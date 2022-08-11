from io import BytesIO, StringIO
import cv2
from anime_face_detector import create_detector
from PIL import Image, ImageDraw, ImageFont
from werkzeug.wsgi import FileWrapper
from flask import Flask, request, Response
import math
import numpy as np

def get_deg(arr):
    rad = math.atan2(arr[3]-arr[1],arr[2]-arr[0])
    PI = math.pi
    deg = (rad*180)/PI
    return deg

detector = create_detector('yolov3', device='cpu')

gg = Image.open('gg.png')

def generate(image_file: BytesIO) -> BytesIO:
    encoded = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    preds = detector(img)
    
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
        color = image.getpixel((points[27][0], points[27][1]+20))
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
        rotated = gg.rotate(-deg, expand=True, center=(0,0))
        resize = math.sqrt((points[10][0] - points[5][0])**2 + (points[10][1] - points[5][1])**2)
        rotated = rotated.resize((int(resize), int(resize*1.12)))
        image.paste(rotated, (int(points[5][0]-(rotated.size[0]*0.9-gg.size[0])), int(points[10][1]-(rotated.size[1]*0.9-gg.size[1]))), rotated)
    
    result = BytesIO()
    image.save(result, 'PNG')
    return result
    

app = Flask(__name__)

@app.post('/generate')
def index():
    if request.files.get('file') is None:
        return "no file", 400
    file = request.files.get('file')
    dst = BytesIO()
    file.save(dst)
    dst.seek(0)
    result = generate(dst)
    result.seek(0)
    result = FileWrapper(result)
    return Response(result, mimetype='image/png', direct_passthrough=True)
    

app.run("0.0.0.0", 8080, debug=False)