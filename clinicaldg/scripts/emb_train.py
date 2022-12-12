import argparse
import json
from ast import literal_eval
from glob import glob

import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

tqdm.pandas()

def parse_args():
    parser = argparse.ArgumentParser(description='Embedding Training Script')
    parser.add_argument('--row_idx', type=int, required=True)
    return parser.parse_args()

def load_all_dfs(row):
    dfs = []
    for csv_file in glob(f"{row.output_dir}/emb-test-*.csv"):
        tmp_df = pd.read_csv(csv_file)
        dfs.append(tmp_df)
        
    return pd.concat(dfs)

def train_models(df, pred_col, pred_vals, fix_col, fix_vals, results_dict, task_type):   
    for fix_val in tqdm(fix_vals):
        # Subset the dataframe to just have this val in the column
        df_fix = df[df[fix_col] == fix_val]
        # Drop this column to avoid any oddities during training
        df_fix = df_fix.drop(columns=fix_col)
        
        # Get just the 2/4 classes that we're trying to 
        for pred_val in tqdm(pred_vals):
            df_pred = df_fix[df_fix[pred_col].isin(pred_val)]
            
            # We have the final dataframe, but we need to create a perfectly balanced 
            # version of it
            grouped = df_pred.groupby(pred_col)
            # print("Count per class:", grouped["emb0"].count())
            min_group_size = grouped.count()["emb0"].min()
            df_bal = grouped.sample(n=min_group_size, random_state=0)
            # print("Count per class after balancing:", df_bal.groupby(pred_col)["emb0"].count())
            
            # Note that we may have a single class remaining in our dataset (if we're doing the baseline 
            # CXP vs CXP prediction, for example). We need to check that and manually change our dataset
            # If that is the case
            df_bal = df_bal.sample(frac=1, random_state=0).reset_index(drop=True)
            
            if len(pred_val) == 1:
                print(f"INFO: SINGLE PRED VAL: {pred_val} for col: {pred_col}... Subsetting through the middle")
                mid_val = len(df_bal) // 2
                
                df_bal.loc[:mid_val, pred_col] = "0"
                df_bal.loc[mid_val:, pred_col] = "1"
            
            # Now lets pass this dataframe into our train method
            acc, coef, intercept = train_model(df_bal, pred_col)
            
            # Store the results in our global dictionary
            results_dict[task_type].append({
                "fix_val": fix_val,
                "pred_val": pred_val,
                "min_group_size": min_group_size,
                "df_size": len(df_bal),
                "acc": acc,
                "coef": coef,
                "intercept": intercept
            })
            
def train_model(df, pred_col, max_iter=5000):
    X, y = df.drop(columns=pred_col), df[pred_col]
    
    model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=max_iter))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    svm_model = model['linearsvc']
    return acc, svm_model.coef_.tolist(), svm_model.intercept_.tolist()

def main():
    df = pd.read_csv("/scratch/rc4499/thesis/ood-generalization/1_baseline_EVAL-wandb_export_2022-11-22T17_52_23.068-05_00.csv")
    print("Whole DF:")
    print(df)
    args = parse_args()

    run = wandb.init(project="ood-generalization",
        job_type="emb_train", 
        entity="basedrhys",
        name=f"Coef: Row {args.row_idx}",
        config=args)

    args = run.config
    row_idx = args.row_idx
    row = df.iloc[row_idx]

    print(f"Running on row: {row}")
    embed_df = load_all_dfs(row)

    output_dir = row["output_dir"]
    print("Outputting JSON to", output_dir)

    # Convert the string representation of the array to an actual array
    tmp_df = embed_df[["env", "targets", "preds"]]
    tmp_df["preds"] = tmp_df["preds"].progress_apply(literal_eval)

    # Split the array out into individual embedding columns
    emb_cols = pd.DataFrame(tmp_df["preds"].to_list(), columns=[f"emb{x}" for x in range(1024)])

    # Concatenate with the metadata columns (env and label)
    ml_df = pd.concat([
        tmp_df.drop(columns="preds").reset_index(drop=True),
        emb_cols.reset_index(drop=True)
    ], axis=1)
    
    # Begin the actual processing
    results_dict = {}
    results_dict["env_pred"] = []
    results_dict["label_pred"] = []

    # Environment Prediction Task
    env_fix_col = "targets"
    env_fix_vals = [0, 1]

    env_pred_col = "env"
    env_pred_vals = [("CXP", ), ("MIMIC", ), ("NIH", ), ("PAD", ), ("CXP","NIH"), ("CXP","PAD"), ("MIMIC","CXP"), ("MIMIC","NIH"), ("MIMIC","PAD"), ("NIH","PAD"), ("CXP", "MIMIC", "NIH", "PAD")]

    # Result is stored in results_dict
    train_models(df=ml_df, 
                 pred_col=env_pred_col,
                 pred_vals=env_pred_vals,
                 fix_col=env_fix_col,
                 fix_vals=env_fix_vals,
                 results_dict=results_dict,
                 task_type="env_pred")
    

    # Label prediction task
    label_fix_col = "env"
    label_fix_vals = ["CXP", "MIMIC", "NIH", "PAD"]

    label_pred_col = "targets"
    label_pred_vals = [(0, 1), (0, ), (1, )]
    
    train_models(df=ml_df,
                 pred_col=label_pred_col,
                 pred_vals=label_pred_vals,
                 fix_col=label_fix_col,
                 fix_vals=label_fix_vals,
                 results_dict=results_dict,
                 task_type="label_pred")
    

    wandb.log(results_dict)
    with open(f"{output_dir}/emb_test_results_coef.json", mode="w") as f:
        json.dump(results_dict, f, indent=True)
    
    print(json.dumps(results_dict, indent=True))

if __name__ == "__main__":
    main()

