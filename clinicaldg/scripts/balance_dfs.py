# singularity exec --nv --overlay /scratch/rc4499/thesis/data/sqf/mimic-cxr-224.sqf:ro \
# --overlay /scratch/rc4499/thesis/data/sqf/CheXpert-v1.0-224.sqf:ro \
# --overlay /scratch/rc4499/thesis/data/sqf/padchest-224.sqf:ro \
# --overlay /scratch/rc4499/thesis/data/sqf/chestxray8-224.sqf:ro \
# --overlay /scratch/rc4499/thesis/pytorch1.7.0-cuda11.0.ext3:ro \
# /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

# conda activate thesis

# python balance_dfs.py --target_disease="Atelectasis" --target_envs="CXP,NIH"

import pandas as pd
import pickle
import argparse
import json

ENVS = ["MIMIC", "CXP", "NIH", "PAD"]
diseases = ["Pneumonia", "Cardiomegaly", "Edema",  "Effusion", 'Atelectasis', 'Pneumothorax', 'Consolidation', "No Finding", "All"]

# target_disease = "Atelectasis"
# target_envs = ["CXP", "NIH"]

def get_prop(df, column):
    num_instances = len(df)
    num_diseased = df[df[column] == 1][column].count()
    return num_diseased / num_instances


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_disease", type=str, required=True)
    parser.add_argument("--target_envs", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def log_dfs(target_envs, all_dfs, is_original, binary_label):
    prefix_str = "orig-" if is_original else ""
    for i, env in enumerate(ENVS):
        log_str = f"{prefix_str}{env}"
        log_splits_for_env(env, all_dfs, log_str, binary_label)
    
    for i, env in enumerate(target_envs):
        log_str = f"{prefix_str}trn_env{i}"
        log_splits_for_env(env, all_dfs, log_str, binary_label)


def log_splits_for_env(env, all_dfs, log_str, binary_label):
    df_trn, df_val, df_test = all_dfs[env]['train'], all_dfs[env]['val'], all_dfs[env]['test']

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
        
    print(json.dumps(env_dict, indent=True), end='\n\n')


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
    # Some basic housekeeping
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
    # Make sure there aren't any values not 0/1
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
        df_diseased_rs = df_diseased.sample(new_num_diseased, replace=True, random_state=seed)
        resampled_df = pd.concat([df_normal, df_diseased_rs])
             
    return resampled_df

    

def main():
    # Get the arguments
    args = parse_args()
    target_disease = args.target_disease
    target_envs = args.target_envs.split(',')
    assert len(target_envs) == 2

    # Parse the folder of raw dataframes
    with open("all_envs_splits_raw/all_dfs_raw.pkl", "rb") as f:
        all_dfs = pickle.load(f)
    """
    all_dfs has structure: {
      "MIMIC": {
           'train': pd.DataFrame(...),
           'val': pd.DataFrame(...),
          'test': pd.DataFrame(...)
      },
      "CXP": {
          ...    
      }
    }
    """
    print(all_dfs.keys())

    log_dfs(target_envs=target_envs, all_dfs=all_dfs, is_original=True, binary_label=target_disease)

    # Figure out which of the two has the larger P(Y = 1)
    df_trn_0, df_trn_1 = all_dfs[target_envs[0]]["train"], all_dfs[target_envs[1]]["train"]
    prop0, prop1 = get_prop(df_trn_0, column=target_disease), get_prop(df_trn_1, column=target_disease)
    target_prop = max(prop0, prop1)
    print(f"Target prop: Using the larger of the two training environments ({prop0}, {prop1}): {target_prop}")

    # Then do the actual balancing of training environments
    for env in target_envs:
        train_df = all_dfs[env]["train"]
        val_df = all_dfs[env]["val"]

        balanced_train_df = balance_proportion(train_df, target_prop, column=target_disease, resample_method="under", seed=args.seed)
        balanced_val_df = balance_proportion(val_df, target_prop, column=target_disease, resample_method="under", seed=args.seed)
        
        all_dfs[env]["train"] = balanced_train_df
        all_dfs[env]["val"] = balanced_val_df

        print(f"New {env} train split prop: {get_prop(all_dfs[env]['train'], column=target_disease)}")
        # print(f"New {train_env} val split prop: {get_prop(self.dfs[train_env]['val'], column=args.binary_label)}")

    log_dfs(target_envs=target_envs, all_dfs=all_dfs, is_original=False, binary_label=target_disease)

    # all_dfs now contains the hospital-balanced dataframes :)

if __name__ == "__main__":
    main()