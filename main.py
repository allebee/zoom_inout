from flask import Flask, Response, render_template

app = Flask(__name__)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    while True:
        # Capture frames from the camera using JavaScript
        frame = yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'


if __name__ == '__main__':
    app.run()
