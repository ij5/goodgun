from io import BytesIO, StringIO
import cv2
from anime_face_detector import create_detector
from werkzeug.wsgi import FileWrapper
from flask import Flask, request, Response, send_file
import math
import numpy as np

def get_deg(arr):
    rad = math.atan2(arr[3]-arr[1],arr[2]-arr[0])
    PI = math.pi
    deg = (rad*180)/PI
    return deg

detector = create_detector('yolov3', device='cpu')

gg = cv2.imread('gg.png', cv2.IMREAD_UNCHANGED)
gg = cv2.cvtColor(gg, cv2.COLOR_BGRA2RGBA)

def generate(image_file: BytesIO) -> bytes:
    encoded = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    preds = detector(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    # for face in preds:
    #     draw.rectangle((face['bbox'][0], face['bbox'][1], face['bbox'][2], face['bbox'][3]), outline=(255, 0, 0), width=5)
    #     x = face['bbox'][0]
    #     y = face['bbox'][1]
    #     for i, point in enumerate(face['keypoints']):
    #         # draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill=(255, 0, 0))
    #         draw.text((point[0], point[1]), str(i), font=ImageFont.truetype('arial.ttf', 10), fill=(255, 0, 0))

    if len(preds) == 0:
        return False

    for face in preds:
        points = face['keypoints']
        color = img[int(points[27][1]), int(points[27][0])+10]
        polygon = np.array([
            [points[0][0], points[0][1]],
            [points[1][0], points[1][1]],
            [points[2][0], points[2][1]],
            [points[3][0], points[4][1]],
            [points[4][0], points[4][1]],
            [points[10][0], points[10][1]],
            [points[9][0], points[9][1]],
            [points[8][0], points[8][1]],
            [points[7][0], points[7][1]],
            [points[6][0], points[6][1]],
            [points[5][0], points[5][1]]
        ], np.int32)
        cv2.fillConvexPoly(img, polygon, color=(int(color[0]), int(color[1]), int(color[2]), 255))
        deg = get_deg([points[0][0], points[0][1], points[4][0], points[4][1]])
        rotated = gg.copy()
        resize = math.sqrt((points[10][0] - points[5][0])**2 + (points[10][1] - points[5][1])**2)
        rotated = cv2.resize(rotated, (int(resize), int(resize*1.12)))
        matrix = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [rotated.shape[0],0], [0, rotated.shape[1]], [rotated.shape[0],rotated.shape[1]]]),
            np.float32([[points[5][0], points[5][1]], [points[10][0], points[10][1]], [points[1][0], points[1][1]], [points[3][0], points[3][1]]]))
        rotated = cv2.warpPerspective(rotated, matrix, (img.shape[1], img.shape[0]))

        alpha = rotated[:, :, 3] / 255.
        for i in range(3):
            pointx, pointy = points[5][:2]
            pointx, pointy = int(pointx), int(pointy)
            img[:, :, i] = (1. - alpha) * img[0:, 0:, i] + alpha * rotated[:, :, i]
    
    
    
    buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))[1]
    
    return buffer.tobytes()
    

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
    if not result:
        return {""}, 400
    return Response(result, mimetype='image/png', direct_passthrough=True)
    

app.run("0.0.0.0", 8080, debug=False)