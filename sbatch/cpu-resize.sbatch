#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=cpu-resize
#SBATCH --output=./out/cpu-resize_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00

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
python resize_imgs.py 1 &
python resize_imgs.py 2 &
python resize_imgs.py 3 &

wait

echo 'Finished waiting for all'
"

cd /vast/rc4499/resized

for a in *; do
	mksquashfs $a $a-800.sqf -processors 8 -keep-as-directory;
done
