#to extract frame and crop face and grt landmark points
import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('../test/assets/aflw-test.jpg')
preds = fa.get_landmarks(input)

preds = fa.get_landmarks_from_directory('../test/assets/')