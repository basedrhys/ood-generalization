#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=cpu-jupyter
#SBATCH --output=./out/cpu-emb_train_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=2:00:00
#SBATCH --array=0-89

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
	/bin/bash -c -c "
source /ext3/env.sh
conda activate thesis
cd /scratch/rc4499/thesis/ood-generalization/ClinicalDG
python clinicaldg/scripts/emb_train.py --row_idx $SLURM_ARRAY_TASK_ID
"
