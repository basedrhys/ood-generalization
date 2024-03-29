#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=gpu-embed
#SBATCH --output=./out/gpu-embed_arr_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --array=0-8%30

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

singularity exec --nv \
	--overlay /scratch/rc4499/thesis/data/sqf/mimic-cxr-224.sqf:ro \
	--overlay /scratch/rc4499/thesis/data/sqf/CheXpert-v1.0-224.sqf:ro \
	--overlay /scratch/rc4499/thesis/data/sqf/padchest-224.sqf:ro \
	--overlay /scratch/rc4499/thesis/data/sqf/chestxray8-224.sqf:ro  \
    --overlay /scratch/rc4499/thesis/pytorch1.7.0-cuda11.0.ext3:ro \
	/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "
source /ext3/env.sh
conda activate thesis
cd /scratch/rc4499/thesis/ood-generalization
python -m clinicaldg.scripts.eval --num_tasks_total 9 --task_id $SLURM_ARRAY_TASK_ID --csv_file /scratch/rc4499/thesis/ood-generalization/1_baseline_EVAL-wandb_export_2022-11-22T17_52_23.068-05_00.csv
"
