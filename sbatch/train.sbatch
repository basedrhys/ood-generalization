#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=train
#SBATCH --output=./out/train_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1

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
bash_scripts/my_cxr.sh train_merged
"
