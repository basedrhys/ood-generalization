#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=cpu-jupyter
#SBATCH --output=./out/cpu-jupyter_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=6:00:00

echo "This script should be run from the top-level of the ood-generalization repo"

port=$(shuf -i 10000-65500 -n 1)

opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -N -f -R $port:localhost:$port"

for((i=1; i<=3; i++)); do
    /usr/bin/ssh $opts $USER@log-$i.nyu.cluster
done

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to greene with command

ssh $USER@greene

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on greene compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

singularity exec --nv \
	--overlay /scratch/wz727/chestXR/data/mimic-cxr.sqsh:ro \
	--overlay /scratch/lhz209/data/chexpert.sqf:ro \
	--overlay /scratch/lhz209/data/padchest.sqf:ro \
	--overlay /scratch/wz727/chestXR/data/chestxray8.sqsh:ro  \
    --overlay /scratch/lhz209/pytorch1.7.0-cuda11.0.ext3:ro \
	/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "
source /ext3/env.sh
conda activate thesis
jupyter-lab --no-browser --port $port --notebook-dir=/scratch/rc4499/thesis
"
