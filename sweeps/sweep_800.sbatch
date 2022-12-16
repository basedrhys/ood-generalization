#!/bin/bash

#SBATCH --job-name=image_size
#SBATCH --output=./out/%j-4_image_size.out
#SBATCH --array=0-33%25
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:rtx8000:1

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

singularity exec --nv \
	--overlay /scratch/rc4499/thesis/data/sqf/mimic-cxr-800.sqf:ro \
	--overlay /scratch/rc4499/thesis/data/sqf/CheXpert-v1.0-800.sqf:ro \
	--overlay /scratch/rc4499/thesis/data/sqf/padchest-800.sqf:ro \
	--overlay /scratch/rc4499/thesis/data/sqf/chestxray8-800.sqf:ro  \
    --overlay /scratch/rc4499/thesis/pytorch1.7.0-cuda11.0.ext3:ro \
	/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "
source /ext3/env.sh
conda activate thesis
cd /scratch/rc4499/thesis/ood-generalization
wandb agent --count 1 basedrhys/ood-generalization/x8sowvlz
"