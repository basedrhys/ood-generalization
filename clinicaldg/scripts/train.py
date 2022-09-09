# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import gc
import os
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import wandb
import json

from clinicaldg import datasets
from clinicaldg import hparams_registry
from clinicaldg import algorithms
from clinicaldg.lib import misc
from clinicaldg.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from clinicaldg.utils import EarlyStopping, has_checkpoint, load_checkpoint, save_checkpoint

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--dataset', type=str, default="CXRBinary")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--es_method', choices = ['train', 'val', 'test'])
    parser.add_argument('--wandb_name', type=str, required=True)

    parser.add_argument('--train_env_0', choices = ['MIMIC', 'CXP', 'NIH', 'PAD'], required=True)
    # Using string None for easier visualisation in W+B
    parser.add_argument('--train_env_1', choices = ['MIMIC', 'CXP', 'NIH', 'PAD', 'none'], default='none')
    parser.add_argument('--val_env', choices = ['MIMIC', 'CXP', 'NIH', 'PAD'], required=True)
    parser.add_argument('--test_env', choices = ['MIMIC', 'CXP', 'NIH', 'PAD'], required=True)

    parser.add_argument('--balance_method', choices = ['none', 'label', 'label+size', 'uniform', 'NURD'], default='none')
    parser.add_argument('--resample_method', choices = ['over', 'under'], default='over')
    parser.add_argument('--num_instances', type=int)
    parser.add_argument('--binary_label', type=str)
    parser.add_argument('--nurd_ratio', type=float)
    parser.add_argument('--img_size', default=224, type=int)

    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--max_steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--delete_model', action = 'store_true', 
        help = 'delete model weights after training to save disk space')
    args = parser.parse_args()

    job_id = os.environ["SLURM_JOB_ID"] if os.environ["SLURM_JOB_ID"] is not None else 99
    args.output_dir = os.path.join(args.output_dir, args.wandb_name + "-" + job_id)
    
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    wandb.init(project="ood-generalization",
                job_type="final_train", 
                entity="basedrhys", 
                config=vars(args),
                name=args.wandb_name)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    wandb.config.update(hparams)
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    ds_class = vars(datasets)[args.dataset]  
    
    # Parse the train/val/test environments from the args
    ds_class.TRAIN_ENVS = [args.train_env_0]
    if args.train_env_1 != 'none':
        ds_class.TRAIN_ENVS.append(args.train_env_1)
    ds_class.VAL_ENV = args.val_env
    ds_class.TEST_ENV = args.test_env

    if args.dataset in vars(datasets):
        dataset = ds_class(hparams, args)
    else:
        raise NotImplementedError
    
    if args.algorithm == 'ERMID': # ERM trained on the training subset of the test env
        TRAIN_ENVS = [ds_class.TEST_ENV]
        VAL_ENV = ds_class.TEST_ENV
        TEST_ENV = ds_class.TEST_ENV
    elif args.algorithm == 'ERMMerged': # ERM trained on merged training subsets of all envs
        TRAIN_ENVS = ds_class.ENVIRONMENTS
        VAL_ENV = ds_class.TEST_ENV  
        TEST_ENV = ds_class.TEST_ENV
    else:
        TRAIN_ENVS = ds_class.TRAIN_ENVS
        VAL_ENV = ds_class.VAL_ENV  
        TEST_ENV = ds_class.TEST_ENV
        
    print("Training Environments: " + str(TRAIN_ENVS))
    print("Validation Environment: " + str(VAL_ENV))
    print("Test Environment: " + str(TEST_ENV))    
  
    train_dss = [dataset.get_torch_dataset([env], 'train', args) for env in TRAIN_ENVS]

    if args.num_instances:
        print(f"INFO: Subsetting training datasets to {args.num_instances} instances")
        for i, ds in enumerate(train_dss):
            ds = torch.utils.data.Subset(ds, np.random.choice(np.arange(len(ds)), min(args.num_instances, len(ds)), replace = False))
            train_dss[i] = ds

    # from collections import Counter
    # for i, env in enumerate(TRAIN_ENVS):
    #     # Get the dataset
    #     ds = train_dss[i]
    #     # Calculate label proportions
    #     labels = []
    #     for j in range(len(ds)):
    #         inst = ds[j]
    #         labels.append(inst[1])
        
    #     print("LABEL NUMBERS")
    #     print(Counter(labels))


    
    train_loaders = [InfiniteDataLoader(
        dataset=i,
        weights=None,
        batch_size=hparams['batch_size'] // len(TRAIN_ENVS), # Want to keep batch size constant
        num_workers=dataset.N_WORKERS)
        for i in train_dss
        ]
    
    if args.es_method == 'val':
        val_ds = dataset.get_torch_dataset([VAL_ENV], 'val', args)
    elif args.es_method == 'train':
        val_ds = dataset.get_torch_dataset(TRAIN_ENVS, 'val', args)
    elif args.es_method == 'test':
        val_ds = dataset.get_torch_dataset([TEST_ENV], 'val', args)
        
    if hasattr(dataset, 'NUM_SAMPLES_VAL'):
        val_ds = torch.utils.data.Subset(val_ds, np.random.choice(np.arange(len(val_ds)), min(dataset.NUM_SAMPLES_VAL, len(val_ds)), replace = False))

    print("Num workers", dataset.N_WORKERS)

    eval_loader = FastDataLoader(
        dataset=val_ds,
        batch_size=hparams['batch_size']*4,
        num_workers=dataset.N_WORKERS)
    
    test_loaders = {env:
        FastDataLoader(
        dataset=dataset.get_torch_dataset([env], 'test', args),
        batch_size=hparams['batch_size']*4,
        num_workers=dataset.N_WORKERS)
     for env in dataset.ENVIRONMENTS   
    }

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(TRAIN_ENVS), hparams, args.dataset, dataset)

    algorithm.to(device)
    
    print("Number of parameters: %s" % sum([np.prod(p.size()) for p in algorithm.parameters()]))

    train_minibatches_iterator = zip(*train_loaders)   
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(i)/hparams['batch_size'] for i in train_dss])

    n_steps = args.max_steps or dataset.MAX_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    
    es = EarlyStopping(patience = dataset.ES_PATIENCE)
    
    if has_checkpoint():
        state = load_checkpoint()
        algorithm.load_state_dict(state['model_dict'])
        algorithm.optimizer.load_state_dict(state['optimizer_dict'])
        [train_loader.sampler.load_state_dict(state['sampler_dicts'][c]) for c, train_loader in enumerate(train_loaders)]
        start_step = state['start_step']
        es = state['es']
        torch.random.set_rng_state(state['rng'])
        print("Loaded checkpoint at step %s" % start_step)
    else:
        start_step = 0        
    
    last_results_keys = None
    pbar = tqdm(range(start_step, n_steps))
    for step in pbar:
        if es.early_stop:
            print("Earling stopping...")
            break
        step_start_time = time.time()
        minibatches_device = [(misc.to_device(xy[0], device), misc.to_device(xy[1], device))
            for xy in next(train_minibatches_iterator)]
        algorithm.train()
        step_vals = algorithm.update(minibatches_device, device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        wandb.log(step_vals)
        pbar.set_postfix({"loss": step_vals["loss"]})

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) and (step > 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # validation
            results.update(dataset.eval_metrics(algorithm, eval_loader, 'es', weights = None, device = device))                        
            wandb.log(results)                    
                
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)    
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")
            
            save_checkpoint(algorithm, algorithm.optimizer, 
                            [train_loader.sampler.state_dict(train_loader._infinite_iterator) for c, train_loader in enumerate(train_loaders)], 
                            step+1, es, torch.random.get_rng_state())
            
            checkpoint_vals = collections.defaultdict(lambda: [])
            
            es(-results['es_' + dataset.ES_METRIC], step, algorithm.state_dict(), os.path.join(args.output_dir, "model.pkl"))            

    # print(f"Num instances after training: {algorithm.num_inst}")
    # print(f"Num positive after training: {algorithm.num_pos}")
    # print(f"Proportion: {algorithm.num_pos / algorithm.num_inst}")

    # print(f"DS 1 - # Inst: {algorithm.num_inst_1}, # Pos: {algorithm.num_pos_1}, Proportion: {algorithm.num_pos_1 / algorithm.num_inst_1}")
    # print(f"DS 2 - # Inst: {algorithm.num_inst_2}, # Pos: {algorithm.num_pos_2}, Proportion: {algorithm.num_pos_2 / algorithm.num_inst_2}")

    # wandb.log({"Training Class Proportion": algorithm.num_pos / algorithm.num_inst})
    # wandb.log({"DS 1 Class Proportion": algorithm.num_pos_1 / algorithm.num_inst_1})
    # wandb.log({"DS 2 Class Proportion": algorithm.num_pos_2 / algorithm.num_inst_2})
    algorithm.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pkl")))
    algorithm.eval()
    
    save_dict = {
        "args": vars(args),
        "model_input_shape": dataset.input_shape,
        "model_num_classes": dataset.num_classes,
        "model_train_domains": TRAIN_ENVS,
        "model_val_domain": VAL_ENV,
        "model_test_domains": TEST_ENV,
        "model_hparams": hparams,
        "es_step": es.step,
        'es_' + dataset.ES_METRIC: es.best_score
    }
    
    final_results = {}         
    for name, loader in test_loaders.items():
        final_results.update(dataset.eval_metrics(algorithm, loader, name, weights = None, device = device))
        
    save_dict['test_results'] = final_results    
    print("Finished final evaluation:")
    print(json.dumps(save_dict, indent=True))
    wandb.log(save_dict)
    torch.save(save_dict, os.path.join(args.output_dir, "stats.pkl"))

    with open(os.path.join(args.output_dir, "stats.json"), mode="w") as f:
        json.dump(save_dict, f)    

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    if args.delete_model:
        os.remove(os.path.join(args.output_dir, "model.pkl"))
