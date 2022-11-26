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

def compute_opt_thresh(target, pred):
    opt_thres = 0
    opt_f1 = 0
    for i in np.arange(0.05, 0.9, 0.01):
        f1 = f1_score(target, pred >= i, average="macro")
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

def binary_clf_metrics(preds, targets, grp, env_name, orig_thresh=None, mask = None):
    if mask is not None:
        preds = preds[mask]
        targets = targets[mask]
        grp = grp[mask]

    opt_thresh = compute_opt_thresh(targets, preds)
    if orig_thresh is not None:
        actual_thresh = orig_thresh
    else:
        actual_thresh = opt_thresh
    print(f"INFO: Optimal threshold: {opt_thresh}, using {actual_thresh}")

    preds_rounded_opt = (preds >= actual_thresh)
    tpr_gap_opt = recall_score(targets[grp], preds_rounded_opt[grp], zero_division = 0) - recall_score(targets[~grp], preds_rounded_opt[~grp], zero_division = 0)
    tnr_gap_opt = tnr(targets[grp], preds_rounded_opt[grp]) - tnr(targets[~grp], preds_rounded_opt[~grp])
    parity_gap_opt = (preds_rounded_opt[grp].sum() / grp.sum()) - (preds_rounded_opt[~grp].sum() / (~grp).sum())    
    phi_opt = matthews_corrcoef(preds_rounded_opt, grp)
    
    if len(np.unique(targets) == 1):
        auroc = -1
    else:
        auroc = roc_auc_score(targets, preds)
    
    return {env_name + '_roc': auroc,
            env_name + '_acc': accuracy_score(targets, preds_rounded_opt),
            env_name + '_prec': precision_score(targets, preds_rounded_opt, average="macro"),
            env_name + '_rec': recall_score(targets, preds_rounded_opt, average="macro"),
            env_name + '_f1': f1_score(targets, preds_rounded_opt, average="macro"),
            env_name + '_prec_bin': precision_score(targets, preds_rounded_opt, average="binary"),
            env_name + '_rec_bin': recall_score(targets, preds_rounded_opt, average="binary"),
            env_name + '_f1_bin': f1_score(targets, preds_rounded_opt, average="binary"),
            env_name + '_mcc': phi_opt,
            env_name + '_opt_thresh': opt_thresh,
            env_name + '_orig_thresh': orig_thresh,
            env_name + '_actual_thresh': actual_thresh,
           }

# env_name + '_tpr_gap': tpr_gap_opt,
# env_name + '_tnr_gap': tnr_gap_opt,
# env_name + '_parity_gap': parity_gap_opt,
# env_name + '_phi': phi_opt,

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

def get_prop(df, column):
    num_instances = len(df)
    num_diseased = df[df[column] == 1][column].count()
    return num_diseased / num_instances

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

"""
https://www.wolframalpha.com/input?i=n1%2F%28n1+%2B+n2%29+%3D+p%2C+solve+for+n2
"""
def calc_rs_n_c1(n, p):
    return int(n * ((1 / p) - 1))

"""
https://www.wolframalpha.com/input?i=n2%2F%28n1+%2B+n2%29+%3D+p%2C+solve+for+n2
"""
def calc_rs_n_c0(n, p):
    return int(-((n * p) / (p - 1)))

def calc_rs_num(n0, n1, p, resample_class):
    if resample_class == 0:
        return calc_rs_n_c1(n1, p)
    else:
        return calc_rs_n_c0(n0, p)

def balance_proportion(orig_df, new_prop, column, resample_method="over", seed=0):
    orig_df = orig_df.fillna(0.0)
    orig_prop = get_prop(orig_df, column)
    assert resample_method in ["over", "under"]
    resample_class = get_resample_class(orig_prop, new_prop, resample_method)
    print(f"Resampling (with seed {seed}) '{column}' via '{resample_method}' on class {resample_class} from {orig_prop} to {new_prop}")
    
    # Estimate the number of items we'll need to resample
    df_normal = orig_df[orig_df[column] == 0.0]
    df_diseased = orig_df[orig_df[column] == 1.0]
    num_normal = len(df_normal)
    num_diseased = len(df_diseased)
    assert num_diseased + num_normal == len(orig_df)

    if resample_method == "over":
        replace=True
    else:
        replace=False

    if resample_class == 0:
        new_num_normal = calc_rs_n_c1(num_diseased, new_prop)
        print(f"Resampling normal samples from {num_normal} to {new_num_normal}")
        df_normal_rs = df_normal.sample(new_num_normal, replace=replace, random_state=seed)
        resampled_df = pd.concat([df_normal_rs, df_diseased])
    else:
        # Resample the diseased class
        new_num_diseased = calc_rs_n_c0(num_normal, new_prop)
        print(f"Resampling diseased samples from {num_diseased} to {new_num_diseased}")
        df_diseased_rs = df_diseased.sample(new_num_diseased, replace=True, random_state=0)
        resampled_df = pd.concat([df_normal, df_diseased_rs])
    #     if resample_class == 0:
    #         new_num_normal = int(num_diseased / new_prop)
    #         print(f"Resampling normal samples from {num_normal} to {new_num_normal}")
    #         df_normal_rs = df_normal.sample(new_num_normal, replace=True, random_state=seed)
    #         resampled_df = pd.concat([df_normal_rs, df_diseased])
    #     else:
    #         # Resample the pneumonia class
    #         new_num_diseased = int(new_prop * num_normal)
    #         print(f"Resampling diseased samples from {num_diseased} to {new_num_diseased}")
    #         df_diseased_rs = df_diseased.sample(new_num_diseased, replace=True, random_state=0)
    #         resampled_df = pd.concat([df_normal, df_diseased_rs])
    # if resample_method == "under":
    #     if resample_class == 0:
    #         new_num_normal = int(num_diseased / new_prop)
    #         print(f"Resampling normal samples from {num_normal} to {new_num_normal}")
    #         df_normal_rs = df_normal.sample(new_num_normal, replace=False, random_state=seed)
    #         resampled_df = pd.concat([df_normal_rs, df_diseased])
    #     else:
    #         # Resample the pneumonia class
    #         new_num_diseased = int(new_prop * num_normal)
    #         print(f"Resampling diseased samples from {num_diseased} to {new_num_diseased}")   
    #         df_diseased_rs = df_diseased.sample(new_num_diseased, replace=False, random_state=0)
    #         resampled_df = pd.concat([df_normal, df_diseased_rs])
             
    return resampled_df

def create_imbalance(df_a: pd.DataFrame, df_b: pd.DataFrame, prop_a: float, disease_col: str, label_val: int, seed: int):
    # Create 0.9 split for column == 0
    df_a_label = df_a[df_a[disease_col] == label_val]
    df_b_label = df_b[df_b[disease_col] == label_val]
    num_total = int(len(df_a_label) / prop_a)
    num_b = num_total - len(df_a_label)
    print(f"LEN A:{label_val} = {len(df_a_label)}")
    print(f"LEN B:{label_val} = {len(df_b_label)}")
    print(f"NUM TOTAL: {num_total}, NUM_B: {num_b}")
    df_b_label = df_b_label.sample(num_b, random_state=seed)

    return df_a_label, df_b_label

def is_diseased(row):
    # diseases = Constants.take_labels[1:]
    return int((row[Constants.take_labels[1:]]).sum() > 0)

def balance_dfs(df_a_0, df_b_0, df_a_1, df_b_1):
    # Balance p(Y=1)=0.5
    # First attach the env label so we can split them back out later
    df_a_0["TMP_ENV"] = "a"
    df_b_0["TMP_ENV"] = "b"
    df_a_1["TMP_ENV"] = "a"
    df_b_1["TMP_ENV"] = "b"

    # Join the dataframes by disease label
    df_0 = pd.concat([df_a_0, df_b_0])
    df_1 = pd.concat([df_a_1, df_b_1])

    # Undersample one of them to match the other
    if len(df_1) > len(df_0):
        df_1 = df_1.sample(n=len(df_0))
    else:
        df_0 = df_0.sample(n=len(df_1))
    
    # Split them back into their constituent parts
    df_a_0 = df_0[df_0["TMP_ENV"] == "a"]
    df_b_0 = df_0[df_0["TMP_ENV"] == "b"]
    df_a_1 = df_1[df_1["TMP_ENV"] == "a"]
    df_b_1 = df_1[df_1["TMP_ENV"] == "b"]

    return df_a_0, df_b_0, df_a_1, df_b_1
            
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
    ES_METRIC = 'f1'
    input_shape = None
    ES_PATIENCE = 10 #  * checkpoint_freq steps
    # TRAIN_ENVS = ['NIH', 'PAD']
    # VAL_ENV = 'MIMIC'
    # TEST_ENV = 'CXP'    
    # NUM_SAMPLES_VAL = 1024*16 # use a subset of the validation set for early stopping
    
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.use_cache = bool(self.hparams['use_cache']) if 'use_cache' in self.hparams else False

        # loads data with random splits
        self.dfs = {}
        for split in cxrConstants.df_paths:
            func = cxrProcess.get_process_func(split)
            df_env = func(pd.read_csv(cxrConstants.df_paths[split]), only_frontal = True)
            df_env["img_exists"] = df_env["path"].apply(img_exists)
            df_env["All"] = df_env.apply(is_diseased, axis=1)
            df_env = df_env[df_env["img_exists"]]
            train_df, valid_df, test_df = cxrProcess.split(df_env)
            self.dfs[split] = {
                'train': train_df,
                'val': valid_df,
                'test': test_df,
            }

        # Log the original length and proportion of the training environments
        self.log_dfs(is_original=True, binary_label=args.binary_label)

        if args.balance_method == "uniform":
            # Store the original train/val sets in a seperate key for later copying
            for i, train_env in enumerate(self.TRAIN_ENVS):
                train_df = self.dfs[train_env]["train"]
                val_df = self.dfs[train_env]["val"]

                self.dfs[train_env]["train_orig"] = copy.deepcopy(train_df)
                self.dfs[train_env]["val_orig"] = copy.deepcopy(val_df)

        if args.balance_method in ["label", "label+size", "uniform"]:
            print("Beginning label balancing")
            # Lets balance the label proportion of all training environments to match the test environment
            test_df = self.dfs[self.TEST_ENV]["train"] # use the train split as its larger and so less variable with seeds
            print("Label balancing... args.label_shift:",args.label_shift)
            if args.label_shift is None or args.label_shift == -1:
                target_prop = get_prop(test_df, column=args.binary_label)
                print(f"Target prop: getting proportion from test df: {target_prop}")
            else:
                target_prop = args.label_shift
                print(f"Target prop: getting proportion from label shift arg: {target_prop}")

            for i, train_env in enumerate(self.TRAIN_ENVS):
                train_df = self.dfs[train_env]["train"]
                val_df = self.dfs[train_env]["val"]

                print(f"\nBalancing {train_env} to match {self.TEST_ENV}")

                balanced_train_df = balance_proportion(train_df, target_prop, column=args.binary_label, resample_method=args.resample_method, seed=args.seed)
                balanced_val_df = balance_proportion(val_df, target_prop, column=args.binary_label, resample_method=args.resample_method, seed=args.seed)
                
                self.dfs[train_env]["train"] = balanced_train_df
                self.dfs[train_env]["val"] = balanced_val_df

                print(f"New {train_env} train split prop: {get_prop(self.dfs[train_env]['train'], column=args.binary_label)}")
                # print(f"New {train_env} val split prop: {get_prop(self.dfs[train_env]['val'], column=args.binary_label)}")
            print()

        # if args.balance_method == "NURD":
        #     assert len(self.TRAIN_ENVS) == 2, "NURD balancing can only be applied with 2 training environments"
        #     env_a, env_b = self.TRAIN_ENVS
        #     splits = ["train", "val", "test"]
        #     labels = [(0, 1), (0, 1), (1, 0)]
        #     for split, label in zip(splits, labels):
        #         df_a, df_b = self.dfs[env_a][split], self.dfs[env_b][split]
        #         df_a_0, df_b_0 = create_imbalance(df_a, df_b, args.nurd_ratio, args.binary_label, label[0], args.seed)
        #         df_b_1, df_a_1  = create_imbalance(df_b, df_a, args.nurd_ratio, args.binary_label, label[1], args.seed)

        #         cols = ["subject_id", "path", args.binary_label]
        #         print("\n\nSPLIT=", split)
        #         print(f"NEW DF_A_{label[0]}=", len(df_a_0))
        #         # print(df_a_0[cols])

        #         print(f"NEW DF_B_{label[0]}=", len(df_b_0))
        #         # print(df_b_0[cols])

        #         print(f"NEW DF_A_{label[1]}=", len(df_a_1))
        #         # print(df_a_1[cols])

        #         print(f"NEW DF_B_{label[1]}=", len(df_b_1))
        #         # print(df_b_1[cols])

        #         df_a_0, df_b_0, df_a_1, df_b_1 = balance_dfs(df_a_0, df_b_0, df_a_1, df_b_1)

        #         print(f"NEW DF_A_{label[0]}=", len(df_a_0))
        #         # print(df_a_0[cols])

        #         print(f"NEW DF_B_{label[0]}=", len(df_b_0))
        #         # print(df_b_0[cols])

        #         print(f"NEW DF_A_{label[1]}=", len(df_a_1))
        #         # print(df_a_1[cols])

        #         print(f"NEW DF_B_{label[1]}=", len(df_b_1))
        #         # print(df_b_1[cols])

        #         df_a = pd.concat([df_a_0, df_a_1]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        #         df_b = pd.concat([df_b_0, df_b_1]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

        #         # # Undersample the bigger one to have the same proportion
        #         # if len(df_a) > len(df_b):
        #         #     df_a = df_a.sample(n=len(df_b), random_state=args.seed)
        #         # else:
        #         #     df_b = df_b.sample(n=len(df_a), random_state=args.seed)


        #         self.dfs[env_a][split] = df_a
        #         self.dfs[env_b][split] = df_b
        
        if args.balance_method in ["label+size", "uniform"]:
            self.balance_size(args.resample_method, args.seed)

        if args.balance_method == "uniform":
            print("INFO: Applying uniform balance method")
            train_ds_size = 1e9
            val_ds_size = 1e9
            for train_env in self.TRAIN_ENVS:
                train_ds_size = min(len(self.dfs[train_env]['train']), train_ds_size)
                val_ds_size = min(len(self.dfs[train_env]['val']), val_ds_size)

            # Uniformly resize all datasets uniformly to have this min size
            for train_env in self.TRAIN_ENVS:
                train_df = self.dfs[train_env]["train_orig"]
                val_df = self.dfs[train_env]["val_orig"]

                train_df = train_df.sample(train_ds_size, random_state=args.seed)
                val_df = val_df.sample(val_ds_size, random_state=args.seed)

                self.dfs[train_env]["train"] = train_df
                self.dfs[train_env]["val"] = val_df

        # Create auxiliary test sets
        self.create_combined_test_sets(args.seed)
        self.create_synthetic_bal_test_sets(args.binary_label, args.seed)
    
        print("\nFINAL CHECK")
        self.log_dfs(is_original=False, binary_label=args.binary_label)

    def log_dfs(self, is_original, binary_label):
        prefix_str = "orig-" if is_original else ""
        for i, env in enumerate(self.ENVIRONMENTS):
            log_str = f"{prefix_str}{env}"
            self.log_splits_for_env(env, log_str, binary_label)
        
        for i, env in enumerate(self.TRAIN_ENVS):
            log_str = f"{prefix_str}trn_env{i}"
            self.log_splits_for_env(env, log_str, binary_label)

        log_str = f"{prefix_str}test_env"
        self.log_splits_for_env(self.TEST_ENV, log_str, binary_label)

    def log_splits_for_env(self, env, log_str, binary_label):
        df_trn, df_val, df_test = self.dfs[env]['train'], self.dfs[env]['val'], self.dfs[env]['test']

        train_split_prop = get_prop(df_trn, column=binary_label)
        train_split_len = len(df_trn)

        val_split_prop = get_prop(df_val, column=binary_label)
        val_split_len = len(df_val)

        test_split_prop = get_prop(df_test, column=binary_label)
        test_split_len = len(df_test)

        env_dict = {
            f"{log_str}-trn_prop": train_split_prop,
            f"{log_str}-trn_len": train_split_len,
            f"{log_str}-val_prop": val_split_prop,
            f"{log_str}-val_len": val_split_len,
            f"{log_str}-test_prop": test_split_prop,
            f"{log_str}-test_len": test_split_len,
        }
        try:
        wandb.config.update(env_dict)
        except Exception as e:
            print("ERROR:")
            print(e)
            wandb.config.update(env_dict, allow_val_change=True)
            
        print(json.dumps(env_dict, indent=True), end='\n\n')

    def create_combined_test_sets(self, seed):
        if len(self.TRAIN_ENVS) < 2:
            return

        train_env_0, train_env_1 = self.TRAIN_ENVS
        test_df_0, test_df_1 = self.dfs[train_env_0]['test'], self.dfs[train_env_1]['test']
        if len(test_df_0) > len(test_df_1):
            # Balance 0 down to 1
            print(f"COMBINING: Balancing env 0 test set ({len(test_df_0)}) down to env 1 ({len(test_df_1)})")
            test_df_0 = test_df_0.sample(len(test_df_1), random_state=seed)
        else:
            # Balance 1 down to 0
            print(f"COMBINING: Balancing env 1 test set({len(test_df_1)}) down to env 0 ({len(test_df_0)})")
            test_df_1 = test_df_1.sample(len(test_df_0), random_state=seed)

        test_df_comb = pd.concat([test_df_0, test_df_1]).sample(frac=1, random_state=seed).reset_index()

        for env in self.ENVIRONMENTS:
            self.dfs[env]['test_combined'] = test_df_comb.copy(deep=True)

    def create_synthetic_bal_test_sets(self, binary_label, seed):
        env = self.TEST_ENV
        print("SYNTHETIC:", env)
        test_df = self.dfs[env]["test"]
        for shift in Constants.LABEL_SHIFTS:
            test_df_balanced = balance_proportion(test_df, shift, resample_method="under", seed=seed, column=binary_label)
            print(f"Final synthetic dataset size: {len(test_df_balanced)}", end="\n\n")
            self.dfs[env][f"test_{shift}"] = test_df_balanced

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
                val_df_1_resampled = val_df_1.sample(min(len(val_df_0), len(val_df_1)), random_state=seed)
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

    def predict_on_set(self, algorithm, loader, device, emb_only=False):
        preds, targets, genders = [], [], []
        algorithm.eval()
        meta_df = None
        with torch.no_grad():
            for x, y, meta in tqdm(loader):
                x = misc.to_device(x, device)
                logits = algorithm.predict(x, emb_only)

                targets += y.detach().cpu().numpy().tolist()
                genders += meta['Sex']
                tmp_df = pd.DataFrame(meta)
                if meta_df is None:
                    meta_df = tmp_df
                else:
                    meta_df = pd.concat([meta_df, tmp_df])

                if not emb_only:
                if y.ndim == 1 or y.shape[1] == 1: # multiclass
                    preds_list = torch.nn.Softmax(dim = 1)(logits)[:, 1].detach().cpu().numpy().tolist()
                else: # multilabel
                    preds_list = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                else:
                    preds_list = logits.detach().cpu().numpy().tolist()

                if isinstance(preds_list, list):
                    preds += preds_list
                else:
                    preds += [preds_list]

        meta_df["preds"] = preds
        meta_df["targets"] = targets
        return np.array(preds), np.array(targets), np.array(genders), meta_df
        
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
        
        opt_thress = [compute_opt_thresh(targets[:, i], preds[:, i])  for i in range(self.num_classes)]
        results.update(self.multilabel_metrics(preds, targets, male, prefix = env_name + '_', suffix = '_thres', thress = opt_thress))
                
        return results    
    
    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']            
        return cxrData.get_dataset(self.dfs, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache)
                
class CXRBinary(CXRBase):
    num_classes = 2
        
    def eval_metrics(self, algorithm, loader, env_name, weights, device, thresh, emb_only=False):
        preds, targets, genders, meta_df = self.predict_on_set(algorithm, loader, device, emb_only)
        male = genders == 'M'        
        if emb_only:
            return meta_df
        else:
        return binary_clf_metrics(preds, targets, male, env_name, orig_thresh=thresh), meta_df
    
    def get_torch_dataset(self, envs, dset, args):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']    
        return cxrData.get_dataset(self.dfs, img_size=args.img_size, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache ,
                                  subset_label = args.binary_label)
       

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

