python -m clinicaldg.scripts.train \
    --algorithm ERM \
    --dataset CXRBinary \
    --output_dir /scratch/rc4499/thesis/output \
    --es_method train \
    --hparams '{"batch_size": 64, "lr": 0.00001}' \
    --wandb_name $1
