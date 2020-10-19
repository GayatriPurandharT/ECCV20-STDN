#np.load(lm_name, allow_pickle=True)
import glob
import os
import shutil
import numpy as np
path = 'data/train/spoof/'
image_folders = os.listdir(path)
# print(image_folders)
for folder in image_folders:
    lm_name= path+folder+'/'+folder+'.npy'
    # print(lm_name)
    try:
        if np.load(lm_name, allow_pickle=True) == None:
            print('removing', path+folder)
            shutil.rmtree(path+folder)
    except ValueError as t:
        continue
# try:
                # fr = meta[random.randint(0, len(meta) - 1)]
            # except:
                # print(_file1, len(meta))
            # im_name = fr
            # lm_name = fr[:-3] + 'npy'
# f= np.load('data/val/spoof/Mask_Trans_61_013/Mask_Trans_61_013.npy')
# print(f)
# if  f == None:
#     print('caught')
