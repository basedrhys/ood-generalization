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
    T.Resize((224, 224)),
                                ])        

csvs = ['processed/padchest.csv', 
'processed/chestxray8.csv', 
'processed/mimic-cxr.csv', 
'processed/CheXpert-v1.0.csv']
# pbar = tqdm(glob.glob("processed/*"))
# for csv in pbar:

    # pbar.set_postfix({"data": csv})
index = int(sys.argv[1])
csv = csvs[index]
print(f"Operating on index: {index}, {csv}")

df = pd.read_csv(csv)
total = len(df)
for p in tqdm(df["path"]):
    # print(f"({i}/{total}")
    # img = Image.open(p).convert('RGBA')
    try: 
        new_dir = f"resized/{os.path.dirname(p)}"
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

    # Save result
    # save_image(final, f"{new_dir}/{os.path.basename(p)}")

print(f"Finished index: {index}, {csv}")