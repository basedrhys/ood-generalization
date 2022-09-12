# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import numpy as np
import pandas as pd
from clinicaldg.lib import misc
import clinicaldg.eicu.Constants as eicuConstants
import clinicaldg.eicu.data as eicuData
import clinicaldg.eicu.Augmentations as eicuAugmentations
import clinicaldg.cxr.Constants as cxrConstants
import clinicaldg.cxr.data as cxrData
import clinicaldg.cxr.Augmentations as cxrAugmentations
import clinicaldg.cxr.process as cxrProcess
from clinicaldg.cxr import Constants
from clinicaldg.scripts.download import mnist_dir
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, matthews_corrcoef
import wandb
import json

from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "ColoredMNIST",
    'eICU',
    'eICUCorrLabel',
    'eICUCorrNoise',
    'eICUSubsampleUnobs',
    'eICUSubsampleObs',
    'CXR',
    'CXRBinary',
    'CXRSubsampleUnobs',
    'CXRSubsampleObs' 
]

def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
    
def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1   

def compute_opt_thres(target, pred):
    opt_thres = 0
    opt_f1 = 0
    for i in np.arange(0.05, 0.9, 0.01):
        f1 = f1_score(target, pred >= i)
        if f1 >= opt_f1:
            opt_thres = i
            opt_f1 = f1
    return opt_thres

def tnr(target, pred):
    CM = confusion_matrix(target, pred, labels=[0, 1])
    
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    return TN/(TN + FP) if (TN + FP) > 0 else 0

def binary_clf_metrics(preds, targets, grp, env_name, mask = None):
    if mask is not None:
        preds = preds[mask]
        targets = targets[mask]
        grp = grp[mask]
    preds_rounded = np.round(preds)
    opt_thres = compute_opt_thres(targets, preds)

    preds_rounded_opt = (preds >= opt_thres)
    tpr_gap_opt = recall_score(targets[grp], preds_rounded_opt[grp], zero_division = 0) - recall_score(targets[~grp], preds_rounded_opt[~grp], zero_division = 0)
    tnr_gap_opt = tnr(targets[grp], preds_rounded_opt[grp]) - tnr(targets[~grp], preds_rounded_opt[~grp])
    parity_gap_opt = (preds_rounded_opt[grp].sum() / grp.sum()) - (preds_rounded_opt[~grp].sum() / (~grp).sum())    
    phi_opt = matthews_corrcoef(preds_rounded_opt, grp)
    
    return {env_name + '_roc': roc_auc_score(targets, preds),
           env_name + '_acc': accuracy_score(targets, preds_rounded_opt),
           env_name + '_prec': precision_score(targets, preds_rounded_opt),
           env_name + '_rec': recall_score(targets, preds_rounded_opt),
           env_name + '_f1': f1_score(targets, preds_rounded_opt),
           env_name + '_tpr_gap': tpr_gap_opt,
           env_name + '_tnr_gap': tnr_gap_opt,
           env_name + '_parity_gap': parity_gap_opt,
           env_name + '_phi': phi_opt,}

class eICUBase():
    '''
    Base hyperparameters:
    eicu_architecture: {MLP, GRU}
    
    '''
    ENVIRONMENTS = ['Midwest', 'West', 'Northeast', 'Missing', 'South']
    TRAIN_PCT = 0.7
    VAL_PCT = 0.1
    MAX_STEPS = 2000
    N_WORKERS = 1
    CHECKPOINT_FREQ = 10
    ES_METRIC = 'roc'
    TRAIN_ENVS = ['Midwest', 'West', 'Northeast']
    VAL_ENV = 'Missing'
    TEST_ENV = 'South'
    num_classes = 2
    input_shape = None
    ES_PATIENCE = 7 # * checkpoint_freq steps
    
    def predict_on_set(self, algorithm, loader, device):
        preds, targets, genders = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = {j: x[j].to(device) for j in x}
                algorithm.eval()
                logits = algorithm.predict(x)

                targets += y.detach().cpu().numpy().tolist()
                genders += x['gender'].cpu().numpy().tolist()
                preds_list = torch.nn.Softmax(dim = 1)(logits)[:, 1].detach().cpu().numpy().tolist()
                if isinstance(preds_list, list):
                    preds += preds_list
                else:
                    preds += [preds_list]
        return np.array(preds), np.array(targets), np.array(genders)

    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        preds, targets, genders = self.predict_on_set(algorithm, loader, device)
        male = genders == 1
        return binary_clf_metrics(preds, targets, male, env_name) # male - female
            
    def get_torch_dataset(self, envs, dset):
        return self.d.get_torch_dataset(envs, dset)
    
        
class eICU(eICUBase): 
    def __init__(self, hparams, args):
        super().__init__()
        self.d = eicuData.AugmentedDataset([], train_pct = eICUBase.TRAIN_PCT, 
                                           val_pct = eICUBase.VAL_PCT)   
        
        
class eICUCorrLabel(eICUBase):    
    '''
    Hyperparameters:
    corr_label_train_corrupt_dist
    corr_label_train_corrupt_mean
    corr_label_val_corrupt
    corr_label_test_corrupt
    '''
    def __init__(self, hparams, args):
        super().__init__()
        self.d = eicuData.AugmentedDataset([eicuAugmentations.AddCorrelatedFeature(hparams['corr_label_train_corrupt_dist'], 
                              hparams['corr_label_train_corrupt_mean'], hparams['corr_label_val_corrupt'], 
                              hparams['corr_label_test_corrupt'], 'corr_label')], 
                       train_pct = eICUBase.TRAIN_PCT, val_pct = eICUBase.VAL_PCT)  
        
        eicuConstants.static_cont_features.append('corr_label')
        

class eICUSubsampleObs(eICUBase):    
    '''
    Hyperparameters:
    subsample_g1_mean
    subsample_g2_mean
    subsample_g1_dist
    subsample_g2_dist
    '''
    def __init__(self, hparams, args):
        super().__init__()
        self.d = eicuData.AugmentedDataset([eicuAugmentations.Subsample(hparams['subsample_g1_mean'], hparams['subsample_g2_mean'],
                                                hparams['subsample_g1_dist'], hparams['subsample_g2_dist'])], 
                       train_pct = eICUBase.TRAIN_PCT, val_pct = eICUBase.VAL_PCT)     
        
        
class eICUSubsampleUnobs(eICUSubsampleObs):    
    '''
    Hyperparameters:
    subsample_g1_mean
    subsample_g2_mean
    subsample_g1_dist
    subsample_g2_dist
    '''
    def __init__(self, hparams, args):
        eicuConstants.static_cat_features.remove('gender')
        super().__init__(hparams, args)
                    
        
class eICUCorrNoise(eICUBase):    
    '''
    Hyperparameters:
    corr_noise_train_corrupt_dist
    corr_noise_train_corrupt_mean
    corr_noise_val_corrupt
    corr_noise_test_corrupt
    corr_noise_std
    corr_noise_feature
    '''
    def __init__(self, hparams, args):
        super().__init__()
        if hparams['corr_noise_feature'] in eicuConstants.ts_cat_features:   # GCS Total     
            eicuConstants.ts_cat_features.remove(hparams['corr_noise_feature'])
            eicuConstants.ts_cont_features.append(hparams['corr_noise_feature'])
                        
        self.d = eicuData.AugmentedDataset([eicuAugmentations.GaussianNoise(hparams['corr_noise_train_corrupt_dist'], hparams['corr_noise_train_corrupt_mean'], 
                                           hparams['corr_noise_val_corrupt'], hparams['corr_noise_test_corrupt'], std = hparams['corr_noise_std'], feat_name = hparams['corr_noise_feature'])], 
                       train_pct = eICUBase.TRAIN_PCT, val_pct = eICUBase.VAL_PCT)          

def get_prop(df, column="Pneumonia"):
    num_instances = len(df)
    num_diseased = df[df[column] == 1][column].count()
    return num_diseased / (num_instances - num_diseased)

def get_resample_class(orig_prop, new_prop, resample_method):
    if new_prop > orig_prop:
        if resample_method == "over":
            return 1
        else:
            return 0
    if new_prop < orig_prop:
        if resample_method == "under":
            return 1
        else:
            return 0

def balance_proportion(orig_df, new_prop, resample_method="over", column="Pneumonia", seed=0):
    orig_df = orig_df.fillna(0.0)
    orig_prop = get_prop(orig_df, column)
    assert resample_method in ["over", "under"]
    resample_class = get_resample_class(orig_prop, new_prop, resample_method)
    print(f"Resampling (with seed {seed}) '{column}' via '{resample_method}' on class {resample_class} from {orig_prop} to {new_prop}")
    
    # Estimate the number of items we'll need to resample
    df_diseased = orig_df[orig_df[column] == 1.0]
    df_normal = orig_df[orig_df[column] == 0.0]
    num_diseased = len(df_diseased)
    num_normal = len(df_normal)
    assert num_diseased + num_normal == len(orig_df)
    if resample_method == "over":
        if resample_class == 0:
            new_num_normal = int(num_diseased / new_prop)
            print(f"Resampling normal samples from {num_normal} to {new_num_normal}")
            df_normal_rs = df_normal.sample(new_num_normal, replace=True, random_state=seed)
            resampled_df = pd.concat([df_normal_rs, df_diseased])
        else:
            # Resample the pneumonia class
            new_num_diseased = int(new_prop * num_normal)
            df_diseased_rs = df_diseased.sample(new_num_diseased, replace=True, random_state=0)
            resampled_df = pd.concat([df_normal, df_diseased_rs])
            print(f"Resampling diseased samples from {num_diseased} to {new_num_diseased}")
            # print("Using imblearn random over sampler")
            # target = orig_df["Pneumonia"] == 1
            # rus = RandomOverSampler(random_state=0, sampling_strategy=new_prop)
            # resampled_df, _ = rus.fit_resample(orig_df, target)
    if resample_method == "under":
        if resample_class == 0:
            new_num_normal = int(num_diseased / new_prop)
            print(f"Resampling normal samples from {num_normal} to {new_num_normal}")
            df_normal_rs = df_normal.sample(new_num_normal, replace=False, random_state=seed)
            resampled_df = pd.concat([df_normal_rs, df_diseased])
        else:
            raise NotImplementedError("Havent done custom undersampling for minority class")
    
    return resampled_df
            
from os.path import exists
def img_exists(path):
    return exists(path)

class CXRBase():
    '''
    Base hyperparameters:
    cxr_augment: {0, 1}
    use_cache: {0, 1}
    '''
    ENVIRONMENTS = ['MIMIC', 'CXP', 'NIH', 'PAD']
    MAX_STEPS = 20000
    N_WORKERS = 2
    CHECKPOINT_FREQ = 100
    ES_METRIC = 'roc'
    input_shape = None
    ES_PATIENCE = 20 #  * checkpoint_freq steps
    # TRAIN_ENVS = ['NIH', 'PAD']
    # TRAIN_ENVS = ['PAD']
    # VAL_ENV = 'MIMIC'
    # TEST_ENV = 'CXP'    
    # TRAIN_ENVS = ['MIMIC', 'CXP']
    # VAL_ENV = 'NIH'
    # TEST_ENV = 'PAD'    
    # NUM_SAMPLES_VAL = 1024*16 # use a subset of the validation set for early stopping
    
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.use_cache = bool(self.hparams['use_cache']) if 'use_cache' in self.hparams else False

        # loads data with random splits
        self.dfs = {}
        for env in cxrConstants.df_paths:
            func = cxrProcess.get_process_func(env)
            df_env = func(pd.read_csv(cxrConstants.df_paths[env]), only_frontal = True)
            df_env["img_exists"] = df_env["path"].apply(img_exists)
            df_env = df_env[df_env["img_exists"]]
            train_df, valid_df, test_df = cxrProcess.split(df_env)
            test_df_balanced = balance_proportion(test_df, 0.5, resample_method=args.resample_method, seed=args.seed)
            print(f"{env} test set indices: {test_df.index[:10]}")
            self.dfs[env] = {
                'train': train_df,
                'val': valid_df,
                'test': test_df,
                'test_bal': test_df_balanced
            }

        # Log the original length and proportion of the training environments
        self.log_dfs(is_original=True)
        if args.balance_method == "none":
            return

        if args.balance_method in ["label", "label+size"]:
            print("Beginning label balancing")
            # Lets balance the label proportion of all training environments to match the test environment
            test_df = self.dfs[self.TEST_ENV]["test"]
            for i, train_env in enumerate(self.TRAIN_ENVS):
                train_df = self.dfs[train_env]["train"]
                val_df = self.dfs[train_env]["val"]

                print(f"\nBalancing {train_env} to match {self.TEST_ENV}")
                balanced_train_df = balance_proportion(train_df, get_prop(test_df), resample_method=args.resample_method, seed=args.seed)
                balanced_val_df = balance_proportion(val_df, get_prop(test_df), resample_method=args.resample_method, seed=args.seed)
                
                self.dfs[train_env]["train"] = balanced_train_df
                self.dfs[train_env]["val"] = balanced_val_df

                print(f"New {train_env} train split prop: {get_prop(self.dfs[train_env]['train'])}")
                print(f"New {train_env} val split prop: {get_prop(self.dfs[train_env]['val'])}")

        
        if args.balance_method == "label+size":
            self.balance_size(args.resample_method, args.seed)
    
        print("FINAL CHECK")
        self.log_dfs(is_original=False)

    def log_dfs(self, is_original):
        for i, train_env in enumerate(self.TRAIN_ENVS):
            prefix_str = "orig-" if is_original else ""
            train_env_prop = get_prop(self.dfs[train_env]['train'])
            train_env_len = len(self.dfs[train_env]['train'])

            val_env_prop = get_prop(self.dfs[train_env]['val'])
            val_env_len = len(self.dfs[train_env]['val'])

            env_dict = {
                f"{prefix_str}trn_env{i}-trn_prop": train_env_prop,
                f"{prefix_str}trn_env{i}-trn_len": train_env_len,
                f"{prefix_str}trn_env{i}-val_prop": val_env_prop,
                f"{prefix_str}trn_env{i}-val_len": val_env_len,
            }
            wandb.config.update(env_dict)
            print(json.dumps(env_dict, indent=True))
            print()

    def balance_size(self, resample_method, seed):
        if len(self.TRAIN_ENVS) == 1:
            return

        train_df_0 = self.dfs[self.TRAIN_ENVS[0]]["train"]
        val_df_0 = self.dfs[self.TRAIN_ENVS[0]]["val"]

        train_df_1 = self.dfs[self.TRAIN_ENVS[1]]["train"]
        val_df_1 = self.dfs[self.TRAIN_ENVS[1]]["val"]

        # Decide the larger of the two datasets
        if resample_method == "under":
            if len(train_df_0) > len(train_df_1):
                # Balance 0 down to 1
                print(f"Balancing env 0 ({len(train_df_0)}) down to env 1 ({len(train_df_1)})")
                train_df_0_resampled = train_df_0.sample(len(train_df_1), random_state=seed)
                val_df_0_resampled = val_df_0.sample(len(val_df_1), random_state=seed)
                self.dfs[self.TRAIN_ENVS[0]]["train"] = train_df_0_resampled
                self.dfs[self.TRAIN_ENVS[0]]["val"] = val_df_0_resampled
            else:
                # Balance 1 down to 0
                print(f"Balancing env 1 ({len(train_df_1)}) down to env 0 ({len(train_df_0)})")
                train_df_1_resampled = train_df_1.sample(len(train_df_0), random_state=seed)
                val_df_1_resampled = val_df_1.sample(len(val_df_0), random_state=seed)
                self.dfs[self.TRAIN_ENVS[1]]["train"] = train_df_1_resampled
                self.dfs[self.TRAIN_ENVS[1]]["val"] = val_df_1_resampled
        else:
            if len(train_df_1) > len(train_df_0):
                # Balance 0 UP to 1
                print(f"Balancing env 0 ({len(train_df_0)}) up to env 1 ({len(train_df_1)})")
                train_df_0_resampled = train_df_0.sample(len(train_df_1), replace=True, random_state=seed)
                val_df_0_resampled = val_df_0.sample(len(val_df_1), replace=True, random_state=seed)
                self.dfs[self.TRAIN_ENVS[0]]["train"] = train_df_0_resampled
                self.dfs[self.TRAIN_ENVS[0]]["val"] = val_df_0_resampled
            else:
                # Balance 1 UP to 0
                print(f"Balancing env 1 ({len(train_df_1)}) up to env 0 ({len(train_df_0)})")
                train_df_1_resampled = train_df_1.sample(len(train_df_0), replace=True, random_state=seed)
                val_df_1_resampled = val_df_1.sample(len(val_df_0), replace=True, random_state=seed)
                self.dfs[self.TRAIN_ENVS[1]]["train"] = train_df_1_resampled
                self.dfs[self.TRAIN_ENVS[1]]["val"] = val_df_1_resampled

    def predict_on_set(self, algorithm, loader, device):
        preds, targets, genders = [], [], []
        with torch.no_grad():
            for x, y, meta in tqdm(loader):
                x = misc.to_device(x, device)
                algorithm.eval()
                logits = algorithm.predict(x)

                targets += y.detach().cpu().numpy().tolist()
                genders += meta['Sex']
                if y.ndim == 1 or y.shape[1] == 1: # multiclass
                    preds_list = torch.nn.Softmax(dim = 1)(logits)[:, 1].detach().cpu().numpy().tolist()
                else: # multilabel
                    preds_list = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                if isinstance(preds_list, list):
                    preds += preds_list
                else:
                    preds += [preds_list]
        return np.array(preds), np.array(targets), np.array(genders) 
        
class CXR(CXRBase):    
    num_classes = len(cxrConstants.take_labels)   
        
    def multilabel_metrics(self, preds, targets, male, prefix, suffix, thress = None):
        if thress is None:
            thress = [0.5] * self.num_classes
            
        tpr_male = np.mean([recall_score(targets[:, i][male], preds[:, i][male] >= thress[i], zero_division=0) for i in range(self.num_classes)])
        tpr_female = np.mean([recall_score(targets[:, i][~male], preds[:, i][~male] >= thress[i], zero_division=0) for i in range(self.num_classes)])
        prev_male = np.mean([(preds[:, i][male] >= thress[i]).sum() / male.sum()  for i in range(self.num_classes)])
        prev_female = np.mean([(preds[:, i][~male] >= thress[i]).sum() / (~male).sum()  for i in range(self.num_classes)])    
        
        return {prefix + 'tpr_gap' + suffix: tpr_male - tpr_female, prefix + 'parity_gap'+ suffix: prev_male - prev_female}
    
    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        preds, targets, genders = self.predict_on_set(algorithm, loader, device)
        male = genders == 'M'
        
        roc = np.mean([roc_auc_score(targets[:, i], preds[:, i]) for i in range(self.num_classes)])
        results = self.multilabel_metrics(preds, targets, male, prefix = env_name+ '_', suffix = '')
        results[env_name+'_roc'] = roc
        
        opt_thress = [compute_opt_thres(targets[:, i], preds[:, i])  for i in range(self.num_classes)]
        results.update(self.multilabel_metrics(preds, targets, male, prefix = env_name + '_', suffix = '_thres', thress = opt_thress))
                
        return results    
    
    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']            
        return cxrData.get_dataset(self.dfs, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache)
                
class CXRBinary(CXRBase):
    num_classes = 2
        
    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        preds, targets, genders = self.predict_on_set(algorithm, loader, device)
        male = genders == 'M'        
        return binary_clf_metrics(preds, targets, male, env_name)
    
    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']    
        return cxrData.get_dataset(self.dfs, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache ,
                                  subset_label = 'Pneumonia')
       

class CXRSubsampleUnobs(CXRBinary):
    '''
    Hyperparameters:
    subsample_g1_mean
    subsample_g2_mean
    subsample_g1_dist
    subsample_g2_dist
    '''
    
    def __init__(self, hparams, args):
        super().__init__(hparams, args)
        self.dfs = cxrAugmentations.subsample_augment(self.dfs, hparams['subsample_g1_mean'], 
                                                                   hparams['subsample_g2_mean'], hparams['subsample_g1_dist'], hparams['subsample_g2_dist'])
        
    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']    
        return cxrData.get_dataset(self.dfs, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache,
                                  subset_label = 'Pneumonia')
   
    
class GenderConcatDataset(Dataset):
    '''
    Wraps a CXR dataset so that the gender information is added to the first output 
    '''
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        x, y, meta = self.ds[idx]
        x_new = {'img': x, 
                 'concat': torch.tensor([meta['Sex'] == 'M']).float()}
        return x_new, y, meta      
    
    
class CXRSubsampleObs(CXRSubsampleUnobs):        
    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']    
        return GenderConcatDataset(cxrData.get_dataset(self.dfs, envs = envs, split = dset, 
                                  imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache,
                                  subset_label = 'Pneumonia'))    

        
class ColoredMNIST():
    '''
    Hyperparameters:
    cmnist_eta
    cmnist_beta
    cmnist_delta   
    
    '''
    ENVIRONMENTS = ['e1', 'e2', 'val']
    TRAIN_PCT = 0.7
    VAL_PCT = 0.1
    MAX_STEPS = 1500
    N_WORKERS = 1
    CHECKPOINT_FREQ = 500 # large value to avoid test env overfitting
    ES_METRIC = 'acc'
    ES_PATIENCE = 10 # no early stopping for CMNIST to avoid test env overfitting
    TRAIN_ENVS = ['e1', 'e2']
    VAL_ENV = 'val'
    TEST_ENV = 'val'
    input_shape = (14*14*2, )
    num_classes = 2
    
    def __init__(self, hparams, args):        
        mnist = MNIST(mnist_dir, train=True, download=True)
        mnist_train = [mnist.data[:50000], mnist.targets[:50000]]
        mnist_val = [mnist.data[50000:], mnist.targets[50000:]]
        idx = np.random.permutation(range(len(mnist_train[1])))
        
        mnist_train[0] = mnist_train[0][idx]
        mnist_train[1] = mnist_train[1][idx]
        
        eta, beta, delta = hparams['cmnist_eta'], hparams['cmnist_beta'], hparams['cmnist_delta']
        
        self.sets = {'e1': {
                'images': mnist_train[0][::2],
                'labels': mnist_train[1][::2]
            },            
            'e2':{
                'images': mnist_train[0][1::2],
                'labels': mnist_train[1][1::2]
            },
            'val':{
                'images': mnist_val[0],
                'labels': mnist_val[1]
            }
           }
    
        self.ps = {'e1': beta + (delta / 2),
              'e2': beta - (delta / 2),
              'val': 0.9}
        
        for s in self.sets:
            imgs = self.sets[s]['images']
            labels = self.sets[s]['labels']
            # 2x subsample for computational convenience
            imgs =  imgs.reshape((-1, 28, 28))[:, ::2, ::2]
            imgs = torch.stack([imgs, imgs], dim = 1)        

            labels = torch_xor((labels < 5).float(),
                                          torch_bernoulli(eta, len(labels)))

            colors = torch_xor(labels, torch_bernoulli(self.ps[s], len(labels)))
            imgs[torch.tensor(range(len(imgs))), (1-colors).long(), :, :] *= 0

            self.sets[s]['images'] = imgs.float()/255.
            self.sets[s]['images'] = self.sets[s]['images'].reshape(self.sets[s]['images'].shape[0], -1)
            self.sets[s]['labels'] = labels.squeeze().long()
            

    def get_torch_dataset(self, envs, dset):
        '''
        envs: a list of region names
        dset: split within envs, one of ['train', 'val', 'test']
        '''
        
        datasets = []
        
        for e in envs:
            xall = self.sets[e]['images']
            if e in self.TRAIN_ENVS:
                if dset == 'train':
                    idx_start, idx_end = 0, int(len(xall) * self.TRAIN_PCT)
                elif dset == 'val':
                    idx_start, idx_end = int(len(xall) * self.TRAIN_PCT), int(len(xall) * (self.TRAIN_PCT + self.VAL_PCT))
                elif dset == 'test':
                    idx_start, idx_end = int(len(xall) * (self.TRAIN_PCT + self.VAL_PCT)), len(xall)
                else:
                    raise NotImplementedError
                    
            elif e == self.VAL_ENV: # on validation environment, use 50% for validation and 50% for test
                if dset == 'val':
                    idx_start, idx_end = 0, int(len(xall) * (0.5))
                elif dset == 'test':
                    idx_start, idx_end = int(len(xall) * (0.5)), len(xall)
                else:
                    raise NotImplementedError
                    
            datasets.append(TensorDataset(xall[idx_start:idx_end], self.sets[e]['labels'][idx_start:idx_end])) 
            
        return ConcatDataset(datasets)      
    
    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        return {env_name + '_acc' : misc.accuracy(algorithm, loader, weights = None, device = device)}
    
    
def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

