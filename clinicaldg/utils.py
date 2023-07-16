import getpass
import os
import torch
from pathlib import Path

class EarlyStopping:
    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_thresh = None
        self.early_stop = False
        self.step = 0

    def __call__(self, val_loss, opt_thresh, step, state_dict, path):  # lower loss is better
        score = -val_loss 

        if self.best_score is None:
            self.best_score = score
            self.best_thresh = opt_thresh
            self.step = step
            save_model(state_dict, path)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"Saving better model with score: {score} and thresh: {opt_thresh} at step {step}")
            save_model(state_dict, path)
            self.best_score = score
            self.best_thresh = opt_thresh
            self.step = step
            self.counter = 0
    
def save_model(state_dict, path):
    torch.save(state_dict, path)              

# functions for checkpoint/reload in case of job pre-emption on our slurm cluster
# will have to customize if you desire this functionality
# otherwise, the training script will still work fine as-is
def save_checkpoint(model, optimizer, sampler_dicts, start_step, es, rng):   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')        
    
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/').exists():        
        torch.save({'model_dict': model.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                    'sampler_dicts': sampler_dicts,
                    'start_step': start_step,
                    'es': es,
                    'rng': rng
        } 
                   , 
                   Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').open('wb')                  
                  )
        
        
def has_checkpoint():
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').exists():
        return True
    return False           

def load_checkpoint():   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    fname = Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt')
    if slurm_job_id is not None and fname.exists():
        return torch.load(fname)       

def get_wandb_name(args):
    train_env_0 = args.train_env_0
    train_env_1 = args.train_env_1 if args.train_env_1 else ""
    balance = args.balance_method
    resample = args.resample_method
    disease = args.binary_label
    # shift = args.label_shift
    ratio = args.nurd_ratio
    img_size = args.img_size
    crop_method = args.crop_method
    return f"({train_env_0},{train_env_1})-disease({disease})-bal({balance},{resample})"
    # return f"({train_env_0},{train_env_1})-test({args.test_env})-shift({shift})-bal({balance},{resample})"
    # return f"new-img_size({img_size})-center_crop({crop_method})"