from io import BytesIO, StringIO
import cv2
from anime_face_detector import create_detector
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from flask import Flask, request, send_file, abort
import math
import numpy as np

def get_deg(arr):
    rad = math.atan2(arr[3]-arr[1],arr[2]-arr[0])
    PI = math.pi
    deg = (rad*180)/PI
    return deg

detector = create_detector('yolov3', device='cpu')

def generate(image_file: BytesIO) -> BytesIO:
    encoded = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    preds = detector(img)

    gg = Image.open('gg.png')
    
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    # for face in preds:
    #     draw.rectangle((face['bbox'][0], face['bbox'][1], face['bbox'][2], face['bbox'][3]), outline=(255, 0, 0), width=5)
    #     x = face['bbox'][0]
    #     y = face['bbox'][1]
    #     for i, point in enumerate(face['keypoints']):
    #         # draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill=(255, 0, 0))
    #         draw.text((point[0], point[1]), str(i), font=ImageFont.truetype('arial.ttf', 10), fill=(255, 0, 0))

    for face in preds:
        points = face['keypoints']
        color = image.getpixel((points[27][0], points[27][1]+5))
        draw.polygon((
            (points[0][0], points[0][1]),
            (points[1][0], points[1][1]),
            (points[2][0], points[2][1]),
            (points[3][0], points[4][1]),
            (points[4][0], points[4][1]),
            (points[10][0], points[10][1]),
            (points[9][0], points[9][1]),
            (points[8][0], points[8][1]),
            (points[7][0], points[7][1]),
            (points[6][0], points[6][1]),
            (points[5][0], points[5][1])
        ), fill=color)

        deg = get_deg([points[0][0], points[0][1], points[4][0], points[4][1]])
        rotated = gg.rotate(-deg)
        resize = math.sqrt((points[4][0] - points[0][0])**2 + (points[4][1] - points[0][1])**2)
        rotated = rotated.resize((int(resize), int(resize*1.12)))
        image.paste(rotated, (int(points[5][0]), int(points[10][1])), rotated)
    
    result = BytesIO()
    image.save(result, 'PNG')
    return result
    

app = Flask(__name__)

@app.post('/generate')
def index():
    if request.files['file'] is None:
        return abort(400)
    file = request.files.get('file')
    print(file.mimetype)
    dst = BytesIO()
    file.save(dst)
    dst.seek(0)
    result = generate(dst)
    print(result.getbuffer().nbytes)
    return send_file(result, mimetype='image/png')
    

app.run("0.0.0.0", 8080, debug=False)