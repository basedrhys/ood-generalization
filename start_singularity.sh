echo "Starting up Singularity container..."

singularity shell --nv \
	--overlay /scratch/wz727/chestXR/data/mimic-cxr.sqsh:ro \
	--overlay /scratch/lhz209/data/chexpert.sqf:ro \
	--overlay /scratch/lhz209/data/padchest.sqf:ro \
	--overlay /scratch/wz727/chestXR/data/chestxray8.sqsh  \
	--overlay /scratch/lhz209/pytorch1.7.0-cuda11.0.ext3:ro \
	/scratch/work/public/singularity/centos-8.2.2004.sif
