#to extract frame and crop face and grt landmark points
import face_alignment
from skimage import io
import os
from shutil import copyfile


def findvideos(inpfile, outpath):
    for dirpath, dirnames, filenames in os.walk("/root/datasets/siwm/"):
        # print('searching for person', i)
        for filename in [f for f in filenames if f.endswith('.mov')]:
            filename = os.path.splitext(filename)[0]
            with open(inpfile) as trainlist:
                for person in trainlist:
                    person = person.strip('\n')
                    if filename == person:
                        copyfile(dirpath+'/'+filename+'.mov', new_testset_path)

        

inpfile = 'testlist.txt'
new_testset_path= '/root/datasets/siwm/test'
os.mkdir(new_testset_path)
findvideos(inpfile, new_testset_path)
#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#input = io.imread('/root/datasets/siwm/Paper/')
#preds = fa.get_landmarks(input)
#preds = fa.get_landmarks_from_directory('../test/assets/')
