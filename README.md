# The Effect of Image Size and Model Capacity on Chest X-Ray Disease Classification Performance

<!-- This is the code for [Paper name](#). -->

## Acknowledgements

The training harness is heavily based on the excellent [ClinicalDG](https://github.com/MLforHealth/ClinicalDG) repo which is in turn a modified version of [DomainBed](https://github.com/facebookresearch/DomainBed).

## To replicate the experiments in the paper:

### Step 0: Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:


### Step 1: Obtaining the Data
See [DataSources.md](DataSources.md) for detailed instructions.

### Step 2: Running Experiments

Experiments can be ran using the same procedure as for the [DomainBed framework](https://github.com/facebookresearch/DomainBed), with a few additional adjustable data hyperparameters which should be passed in as a JSON formatted dictionary.

For example, to train a single model:
```
python -m clinicaldg.scripts.train\
	--algorithm ERM\
	--dataset eICUSubsampleUnobs\
	--es_method val\
	--hparams  '{"eicu_architecture": "GRU", "eicu_subsample_g1_mean": 0.5, "eicu_subsample_g2_mean": 0.05}'\
	--output_dir /path/to/output
```

A detailed list of `hparams` available for each dataset can be found [here](hparams.md).

We provide the bash scripts used for our main experiments in the `bash_scripts` directory. You will likely need to customize them, along with the launcher, to your compute environment.

## W+B Sweeps

This codebase heavily utilises [W+B](https://wandb.ai/site) to run experiments, both for tracking and recording results, and running experiments via the [Sweeps](https://docs.wandb.ai/guides/sweeps) feature (along side Slurm arrays).

The process for this is as follows:

* Define your sweep hyperparameters in a `.yaml` file (e.g., [this YAML file for image size experimentation](./sweeps/4_image_model_size.yaml))
* Start the sweep: `wandb sweep <yaml filename>`
* Create your Slurm array script (e.g., [sweep.sbatch](./sweeps/sweep.sbatch)) -- the key parameter is the `--array=` feature, which should be set to `0-<num parameter configs>`
* Start the Slurm array job: `sbatch sweeps/sweep.sbatch`
* Sit back and watch GPUs go brrrr

## Loading the Datasets

The following steps walkthrough the process for loading datasets and applying hospital-label balancing

* Main entrypoint: `clinicaldg/scripts/train.py`
* Corresponding W+B `.yaml` file for hospital-label balancing: `sweeps/2d_nurd_fix.yaml`
  * The key parameter is :`--balance_resample "label_notest,under"`, which does label-balancing with no target test hospital (i.e., label balance the two hospitals to each other), via undersampling. `"hospital_label,under"` would be a more appropriate name, in hindsight.
  * The `--test_env` is not used but just provided as a placeholder
  * This file gives a range of example invocations of the script, e.g. the following will train a model for **Pneumonia** prediction on **MIMIC** and **CXP**, balancing them such that `P(Y = 1 | Hospital = MIMIC) == P(Y = 1 | Hospital = CXP)`:

    ```bash
    python ./clinicaldg/scripts/train.py --max_steps 20001 --train_envs MIMIC,CXP --test_env MIMIC --balance_resample "label_notest,under" --binary_label Pneumonia
    ```
  
* The codebase is very generalized (to be used in both the eICU task and CXR classification) so has a lot of code to sift through. The path to dataset loading (and ultimately balancing) is from `train.py`, then through:
  * `dataset = ds_class(hparams, args)`, which invokes...
  * `CXRBase __init__()` (in `clinicaldg/datasets.py`)
  * Lines `411` to `447` in `clinicaldg/datasets.py` do the actual hospital-label balancing

If wanting to use this data within another codebase, one could save the train/val/test DFs to CSV files after processing is finished, i.e., at line `496`; these contain the file paths / labels and can be easily loaded into another python project.

### Dataloader

The PyTorch dataset can be found in `clinicaldg/cxr/data.py`, and the `get_dataset()` function in that file also.

The other key class is the `InifiniteDataLoader` used in `train.py` Line `303`. One of these is created for each hospital we're training on, and equal sized batches are sampled from each at every iteration.
