#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=cpu-resize
#SBATCH --output=./out/cpu-resize_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=18:00:00

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

singularity exec --nv \
	--overlay /scratch/rc4499/thesis/data/sqf/mimic-cxr-comb.sqsh \
	--overlay /scratch/rc4499/thesis/data/sqf/chexpert.sqf \
	--overlay /scratch/rc4499/thesis/data/sqf/padchest.sqf \
	--overlay /scratch/rc4499/thesis/data/sqf/chestxray8.sqsh  \
    --overlay /scratch/rc4499/thesis/pytorch1.7.0-cuda11.0.ext3:ro \
	/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "
source /ext3/env.sh
conda activate thesis
python resize_imgs.py 0 &
python resize_imgs.py 2 &

wait

echo 'Finished waiting for all'
"

cd resized

for a in *; do
	mksquashfs $a $a-224.sqf -processors 4 -keep-as-directory;
done
