import cv2
from anime_face_detector import create_detector
from PIL import Image, ImageDraw, ImageFont

detector = create_detector('yolov3', device='cpu')
img = cv2.imread('test.jpg')
preds = detector(img)
print(len(preds[0]['keypoints']))
print(preds[0])

image = Image.fromarray(img)