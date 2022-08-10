import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw, ImageFont
from CFA import CFA
import math
# import animeface


def get_deg(arr):
    rad = math.atan2(arr[3]-arr[1],arr[2]-arr[0])
    PI = math.pi
    deg = (rad*180)/PI
    return deg

# param
num_landmark = 24
img_width = 128
checkpoint_name = 'checkpoint_landmark_191116.pth.tar'
input_img_name = input("이미지 이름> ")

# detector
face_detector = cv2.CascadeClassifier('lbpcascade_animeface.xml')
landmark_detector = CFA(output_channel_num=num_landmark + 1, checkpoint_name=checkpoint_name)

# transform
normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
train_transform = [transforms.ToTensor(), normalize]
train_transform = transforms.Compose(train_transform)

# input image & detect face
img = cv2.imread(input_img_name)
faces = face_detector.detectMultiScale(img)
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img)

gg = Image.open('gg.png')
img.paste(gg, (0, 0), gg)

for x_, y_, w_, h_ in faces:

    # adjust face size
    x = max(x_ - w_ / 8, 0)
    rx = min(x_ + w_ * 9 / 8, img.width)
    y = max(y_ - h_ / 4, 0)
    by = y_ + h_
    w = rx - x
    h = by - y

    # draw result of face detection
    draw.rectangle((x, y, x + w, y + h), outline=(0, 0, 255), width=3)
    print(w, h)

    # transform image
    img_tmp = img.crop((x, y, x+w, y+h))
    img_tmp = img_tmp.resize((img_width, img_width), Image.BICUBIC)
    img_tmp = train_transform(img_tmp)
    img_tmp = img_tmp.unsqueeze(0)

    # estimate heatmap
    heatmaps = landmark_detector(img_tmp)
    heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

    def get(i: int):
        heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
        landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
        landmark_y = landmark[0] * h / img_width
        landmark_x = landmark[1] * w / img_width
        return (landmark_x, landmark_y)

    size = (w/20, h/20)

    colorpix = get(9)
    color = img.getpixel((x+colorpix[0]+size[0], y+colorpix[1]))
    
    points = []
    landmark_x, landmark_y = get(0)
    points.append((x+landmark_x+5, y+landmark_y))
    landmark_x, landmark_y = get(1)
    points.append((x+landmark_x, y+landmark_y-20))
    landmark_x, landmark_y = get(2)
    points.append((x+landmark_x-10, y+landmark_y))
    landmark_x, landmark_y = get(8)
    points.append((x+landmark_x, y+landmark_y))
    landmark_x, landmark_y = get(7)
    points.append((x+landmark_x, y+landmark_y))
    landmark_x, landmark_y = get(6)
    points.append((x+landmark_x, y+landmark_y))
    landmark_x, landmark_y = get(5)
    points.append((x+landmark_x, y+landmark_y))
    landmark_x, landmark_y = get(4)
    points.append((x+landmark_x, y+landmark_y))
    landmark_x, landmark_y = get(3)
    points.append((x+landmark_x, y+landmark_y))
    # calculate landmark position
    for i in range(num_landmark):
        # heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
        # landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
        # landmark_y = landmark[0] * h / img_width
        # landmark_x = landmark[1] * w / img_width
        # landmark_x, landmark_y = get(i)
        # if i == 0:
        #     points.append((x+landmark_x+5, y+landmark_y))
        # if i == 1:
        #     points.append((x+landmark_x, y+landmark_y-20))
        # if i == 2:
        #     points.append((x+landmark_x-10, y+landmark_y))
        # if i == 8:
        #     points.append((x+landmark_x, y+landmark_y))
        # if i == 7:
        #     points.append((x+landmark_x, y+landmark_y))
        # if i == 6:
        #     points.append((x+landmark_x, y+landmark_y))
        # if i == 5:
        #     points.append((x+landmark_x, y+landmark_y))
        # if i == 4:
        #     points.append((x+landmark_x, y+landmark_y))
        # if i == 3:
        #     points.append((x+landmark_x, y+landmark_y))
            
        
                
        # draw landmarks
        landmark_x, landmark_y = get(i)

        font = ImageFont.truetype('arial.ttf', 10)
        draw.text((x+landmark_x-5, y+landmark_y-5), str(i), (255, 0, 0), font)
        # draw.ellipse((x + landmark_x - 2, y + landmark_y - 2, x + landmark_x + 2, y + landmark_y + 2), fill=(255, 0, 0))
    
    draw.polygon((points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7], points[8]), fill=color, width=30, outline=color)
    
    firstx, firsty = get(0)
    secondx, secondy = get(2)
    deg = get_deg([x+firstx, y+firsty, x+secondx, y+secondy])
    new = gg.rotate(-int(deg))
    resize = math.sqrt(((x+secondx)-(x+firstx))**2+((y+secondy)-(y+firsty))**2)
    new = new.resize((int(resize), int(resize * 1.12)))
    img.paste(new, (int(x+firstx), int(y+get(6)[1])), new)
    
        
    
# output image
img.save('output.png')