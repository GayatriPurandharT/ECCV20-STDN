#to extract frame and crop face and grt landmark points
import face_alignment
from skimage import io
import os
import subprocess
from shutil import copyfile
from mtcnn import MTCNN
import cv2

def extract_frames(input_path, name, outpath):
    cmd = "ffmpeg -i "+ input_path+name+".mov -vf \"select=not(mod(n\,5))\" -vsync vfr -q:v 2 "+outpath+name+'/'+name+"_%03d.jpg"
    subprocess.run([cmd], shell=True, executable = '/bin/bash')
def findvideos(inpfile, outpath):
    for dirpath, dirnames, filenames in os.walk("/root/datasets/siwm/"):
        # print('searching for person', i)
        for filename in [f for f in filenames if f.endswith('.mov')]:
            filename = os.path.splitext(filename)[0]
            with open(inpfile) as trainlist:
                for person in trainlist:
                    person = person.strip('\n')
                    if filename == person:
                        # print(dirpath)
                        if not os.path.isdir(outpath+person+'/'):
                            os.mkdir(outpath+person+'/')      
                        # copyfile(dirpath+'/'+filename+'.mov', outpath)
                        extract_frames(dirpath+'/',person, outpath)

        

def crop face():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    input = io.imread('/root/datasets/siwm/Paper/')
    preds = fa.get_landmarks(input)
    

inpfile = 'testlist.txt'                        
new_testset_path= '/root/datasets/siwm/test/'
if not os.path.isdir(new_testset_path):
    os.mkdir(new_testset_path)
findvideos(inpfile, new_testset_path)

#input = io.imread('/root/datasets/siwm/Paper/')
#preds = fa.get_landmarks(input)
#preds = fa.get_landmarks_from_directory('../test/assets/')
