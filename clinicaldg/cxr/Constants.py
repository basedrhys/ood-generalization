import os

#-------------------------------------------
image_paths = {
    'MIMIC': '/mimic-cxr', # MIMIC-CXR
    'CXP': '/CheXpert-v1.0', # CheXpert
    'NIH': '/chestxray8', # ChestX-ray8
    'PAD': '/padchest', # PadChest
}

meta_paths = {
    'MIMIC': '/scratch/rc4499/thesis/data/mimic-cxr', # MIMIC-CXR
    'CXP': '/scratch/rc4499/thesis/data/chexpert', # CheXpert
    'NIH': '/scratch/rc4499/thesis/data/chestxray8', # ChestX-ray8
    'PAD': '/scratch/rc4499/thesis/data/padchest', # PadChest
}

cache_dir = '/scratch/rc4499/thesis/cache'

#-------------------------------------------

df_paths = {
    dataset: os.path.join(meta_paths[dataset], 'clinicaldg', f'preprocessed.csv')
    for dataset in meta_paths 
}

take_labels = ['No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]

IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

LABEL_SHIFTS = [0.1, 0.2, 0.5, 0.8, 0.9]