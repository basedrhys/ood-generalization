#!/bin/bash

#SBATCH --job-name=image_size
#SBATCH --output=./out/%j-3b_label_balance_fix.out
#SBATCH --array=0-16%25
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=5:30:00
#SBATCH --gres=gpu:rtx8000:1

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
wandb agent --count 1 basedrhys/ood-generalization/azww6r8l
"