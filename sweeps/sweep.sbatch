#!/bin/bash

#SBATCH --job-name=grid_search
#SBATCH --output=./out/%j-test-grid_search.out
#SBATCH --array=0-1200%50
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=4:00:00
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
cd /scratch/rc4499/thesis/ood-generalization/ClinicalDG
wandb agent --count 1 basedrhys/ood-generalization/if4t5xx2
"