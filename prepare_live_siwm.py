import shutil
#to extract frame and crop face and grt landmark points
from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage import io
import os
import face_alignment
import subprocess
from PIL import Image
from shutil import copyfile
import cv2
import numpy

def extract_frames(input_path, folder, name):
    cmd = "ffmpeg -i "+ input_path+folder+name+'/'+name+".mov -vf \"select=not(mod(n\,5))\" -vsync vfr -q:v 2 "+input_path+folder+name+'/'+name+"_%03d.png"
    subprocess.run([cmd], shell=True, executable = '/bin/bash')
    
def mtcnn_caller(path, image_folder,image, output):
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256,margin=0)
    # img = cv2.imread(path+image_folder+image)
    # img = Image.open(path+image_folder+image+'/'+image+'.png')
    img = Image.open(path+image_folder+image)
    print('taken input from',path+image_folder+image)
    # detector = MTCNN()
    # faces=detector.detect_faces(img)
    faces,_= mtcnn.detect(img)
    if faces is not None:
    # try:
    #     img = Image.open(path+image_folder+image)
    # # do stuff
    # except IOError:
    # # filename not an image file
    #     print('not image')
    #     img = cv2.imread(path+image_folder+image)
        
        # print('1&&&&&&_____________________&&&&&&&&&&&&&&&', img)
        if img is not None:
            # Get cropped and prewhitened image tensor
            image = image.strip('.png')
            img_cropped = mtcnn(img, save_path=output+image+'/'+image+'.png')
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            input = cv2.imread(output+image+'/'+image+'.png')
            # image = image.strip('.png')
            # print('2&&&&&&_____________________&&&&&&&&&&&&&&&', input)
            try:
                # image = image.strip('.jpg')
                preds = fa.get_landmarks(input)
                numpy.save(output+image+'/'+image, preds)
            except IndexError as error:
                print(error)
            print('check output at',output+image+'/'+image+'.png')

def get_live_videos(path, image_folder, image):
    if image.endswith('.mov'):
                new_folder= image.strip('.mov')
                os.mkdir(path+image_folder+'/'+new_folder)
                shutil.move(path+image_folder+'/'+image, path+image_folder+'/'+new_folder+'/')

if __name__ =='__main__':

    # inpfile = 'trainlist.txt'                        
    path= '/root/datasets/siwm/Live/Train/'
    # if not os.path.isdir(new_testset_path):
        # os.mkdir(new_testset_path)
    outpath = '/root/datasets/siwm/train/live/'
    # findvideos(inpfile, new_testset_path)
    image_folders = os.listdir(path)
    # print(image_folders)
    for image_folder in image_folders:
        # print(image_folder)
        if not image_folder.endswith('.face'):
            images = os.listdir(path+image_folder)
            # print(images)
            for image in images:

                # get_live_videos(path, image_folder,image)
                # if not image.endswith('.face'):
                #     extract_frames(path, image_folder+'/', image)

                # frames = os.listdir(path+image_folder+'/'+image)
                # for frame in frames:
                if not image.endswith('.mov'):
                    mtcnn_caller(path, image_folder+'/', image, outpath)


                # mtcnn_caller(path, image_folder+'/', image, outpath,)