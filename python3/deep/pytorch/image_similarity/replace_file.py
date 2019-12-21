import os
import sys
import shutil
from tqdm import tqdm


dir = "/Users/yuji.kamiya/Desktop/captures_0/node/cropaaaaaaaaaa"

for file in tqdm(os.listdir(dir)):
    splited = file.rsplit('_', 1)
    new_file = '@'.join(splited)
    shutil.move(os.path.join(dir, file), os.path.join(dir, new_file))
