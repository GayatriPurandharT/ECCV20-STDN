from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import face_alignment
from skimage import io
import numpy

def mtcnn_caller(path, image_folder,image):
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256,margin=0)
    img = Image.open(path+image_folder+image)
    if img
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=path+'spoof/'+image)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    input = io.imread(path+'spoof/'+image)
    image = image.strip('.jpg')
    preds = fa.get_landmarks(input)
    numpy.save(path+'spoof/'+image, preds)




path = '/root/datasets/siwm/test/'
image_folders = os.listdir(path)
# print(image_folders)
for image_folder in image_folders:
    # print(image_folder)
    images = os.listdir(path+image_folder)
    # print(images)
    for image in images:
        # print(image)
#         print(path+'/'+image_folder+'/'+image)
        mtcnn_caller(path, image_folder+'/',image)

