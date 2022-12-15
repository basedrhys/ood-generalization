# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
from copy import deepcopy
import json
import os
import random
import sys
import time
import gc
import os
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
import wandb
import json

sys.path.append("/scratch/rc4499/thesis/ood-generalization/ClinicalDG")

from clinicaldg import datasets
from clinicaldg import hparams_registry
from clinicaldg import algorithms
from clinicaldg.lib import misc
from clinicaldg.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from clinicaldg.utils import EarlyStopping, has_checkpoint, load_checkpoint, save_checkpoint, get_wandb_name
from clinicaldg.cxr.Constants import LABEL_SHIFTS

torch.multiprocessing.set_sharing_strategy('file_system')

def run_testing(dataset, split_name, environments, args, algorithm, device, bs, thresh):
    print(f"TEST: Running on split: {split_name} for environments: {environments}")
    test_loader = {env:
        FastDataLoader(
        dataset=dataset.get_torch_dataset([env], split_name, args),
        batch_size=bs,
        num_workers=dataset.N_WORKERS,
        shuffle=False)
     for env in environments   
    }
    final_results = {}         
    for name, loader in test_loader.items():
        print("Evaluating:", name)
        meta_df = dataset.eval_metrics(algorithm, loader, name, weights = None, device = device, thresh = thresh, emb_only=True)
        # final_results.update(eval_metrics)
        
        meta_df.to_csv(f"{args.output_dir}/emb-{split_name}-{name}.csv")

    del test_loader
    gc.collect()

    return final_results

def run_eval(args):
    alg_type = "ERM"
    dataset_type = "CXRBinary"
    model_type = "densenet121"
     # print('Args:')
    # for k, v in sorted(args.items()):
    #     print('\t{}: {}'.format(k, v))

    # if args.balance_method is None and not (args.label_shift == -1 or args.label_shift is None):
    #     print("Unneeded training config")
    #     print(args.balance_method, args.label_shift)
    #     exit()

    hparams = hparams_registry.random_hparams(alg_type, dataset_type,
        misc.seed_hash(0, 0))

    # wandb.config.update(hparams)
    # wandb.config.update({"slurm_id": job_id})
    # print('HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"DEVICE: {device}")
        
    ds_class = vars(datasets)[dataset_type]  
    
    # Parse the train/val/test environments from the args
    ds_class.TRAIN_ENVS = ["MIMIC"]
    ds_class.VAL_ENV = "MIMIC"
    ds_class.TEST_ENV = "MIMIC"

    dataset = ds_class(hparams, args)
    
    TRAIN_ENVS = ds_class.TRAIN_ENVS
    VAL_ENV = ds_class.VAL_ENV  
    TEST_ENV = ds_class.TEST_ENV
        
    print("Training Environments: " + str(TRAIN_ENVS))
    print("Validation Environment: " + str(VAL_ENV))
    print("Test Environment: " + str(TEST_ENV))    

    print("Running eval on row:")
    print(args)

    algorithm_class = algorithms.get_algorithm_class(alg_type)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(TRAIN_ENVS), hparams, dataset_type, dataset, model_type)

    algorithm.to(device)
    algorithm.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pkl"), map_location=device))
    algorithm.eval()

    run_testing(
        dataset=dataset,
        split_name="test",
        environments=dataset.ENVIRONMENTS,
        thresh=args.es_opt_thresh,
        args=args,
        algorithm=algorithm,
        device=device,
        bs=hparams["batch_size"]*4)
    
    print("Finished final evaluation:")

    del algorithm
    del dataset
    gc.collect()
    return None
    # print(json.dumps(save_dict, indent=True))
    # wandb.log(save_dict)
    # torch.save(save_dict, os.path.join(args.output_dir, "stats.pkl"))

    # with open(os.path.join(args.output_dir, "stats.json"), mode="w") as f:
    #     json.dump(save_dict, f)    

    # with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    #     f.write('done')

    # if args.delete_model:
    #     os.remove(os.path.join(args.output_dir, "model.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--dataset', type=str, default="CXRBinary")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--wandb_name', type=str)

    parser.add_argument('--model_type', type=str, default="densenet121")
    
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--task_id', type=int)
    parser.add_argument('--num_tasks_total', type=int)

    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--job_id', type=int)
    args = parser.parse_args()

    job_id = os.environ["SLURM_JOB_ID"] if os.environ["SLURM_JOB_ID"] is not None else "9999"
    args.job_id = job_id
    print(f"SLURM JOB ID:", args.job_id)

    wandb.init(project="ood-generalization",
                job_type="eval", 
                entity="basedrhys", 
                config=args)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    df = pd.read_csv(args.csv_file)

    print("TASK ID:", args.task_id)
    if args.task_id is not None:
        # We're splitting the task into subsets, so lets find our slice of the total DF
        slice_size = len(df) // args.num_tasks_total
        slice_start = args.task_id * slice_size
        slice_end = (args.task_id + 1) * slice_size

        df = df.iloc[slice_start:slice_end]

    print(df)

    df.progress_apply(run_eval, axis=1)