import pandas as pd
import glob
from tqdm import tqdm
from skimage import io
from torchvision.utils import save_image
import torch
from PIL import Image
import torchvision.transforms as T
import os
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

input_transform = T.Compose([
    T.Resize((800, 800)),
                                ])        

csvs = ['processed/PAD.csv', 
'processed/NIH.csv', 
'processed/MIMIC.csv', 
'processed/CXP.csv']

index = int(sys.argv[1])
csv = csvs[index]
print(f"Operating on index: {index}, {csv}")

df = pd.read_csv(csv)
total = len(df)
for p in tqdm(df["path"]):
    # print(f"({i}/{total}")
    # img = Image.open(p).convert('RGBA')
    try: 
        new_dir = f"/vast/rc4499/resized/{os.path.dirname(p)}"
        new_filepath = f"{new_dir}/{os.path.basename(p)}"
        if os.path.exists(new_filepath):
            continue

        img = Image.open(p)
        # Transform image
        final = input_transform(img)

        os.makedirs(new_dir, exist_ok=True)
        final.save(new_filepath)
    except Exception as e:
        print()
        print(p)
        print(e)

print(f"Finished index: {index}, {csv}")