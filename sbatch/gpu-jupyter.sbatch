#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue


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
conda activate dls
jupyter-lab --no-browser --port $port --notebook-dir=$(pwd)
"

