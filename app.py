from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)
    model = YOLO('yolov8n.pt')
    while True:
        success, frame = camera.read()
        results = model.predict(frame, device='cpu')
        frame = results[0].plot()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
if __name__ == "__main__":
    app.run(debug=True)
            
