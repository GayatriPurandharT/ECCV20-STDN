#np.load(lm_name, allow_pickle=True)
import glob
import os
import shutil
import numpy as np
path = 'data/test/live/'
image_folders = os.listdir(path)
# print(image_folders)
for folder in image_folders:
    lm_name= path+folder+'/'+folder+'.npy'
    # print(lm_name)
    if os.path.isfile(lm_name)==True:
        try:
            if np.load(lm_name, allow_pickle=True) == None:
                print('removing', path+folder)
                shutil.rmtree(path+folder)
        except ValueError as t:
            continue
    else: 
        shutil.rmtree(path+folder)
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
