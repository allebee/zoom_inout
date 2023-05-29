import cv2
import numpy as np
import cv2
import dlib
import numpy as np
from imutils import face_utils
from flask import Flask, Response

app = Flask(__name__)

# Define the oval region for face positioning
global width, height
width = 300
height = 400
oval_center = (320, 240)  # Default center for 640x480 camera resolution
text = "Please zoom in"


def capture_frames(center, text):
    global width, height
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Set the camera resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize the face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()  # defauld dlib face detector
    predictor = dlib.shape_predictor(
        'model_needed _zoom.dat')
    count = 0
    while True:
        # Capture a frame from the camera
        ret, frame1 = cap.read()
        frame = cv2.flip(frame1, 1)

        # Check if the frame was successfully captured
        if not ret:
            print('Failed to capture frame from camera.')
            break

        # Detect faces using the face detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        axes = (width//2, height//2)
        inner_width = width//1.5
        inner_height = height//1.5
        inner_center = center
        inner_axes = (int(inner_width//2), int(inner_height//2))

        # Check if a face is detected within the oval region
        face_in_oval = False
        for face in faces:
            # Get the facial landmarks for the current face
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            landmark_points = landmarks[:3]
            imagePoints = landmarks[16:18]
            for (x, y) in landmark_points:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            distances_inner = ((landmark_points[:, 0]-inner_center[0])/inner_axes[0])**2 + (
                (landmark_points[:, 1]-inner_center[1])/inner_axes[1])**2

            distances_outer = ((landmark_points[:, 0]-center[0])/axes[0])**2 + (
                (landmark_points[:, 1]-center[1])/axes[1])**2

            if np.all(distances_inner >= 1) and np.all(distances_outer <= 1):
                face_in_oval = True
        # Draw the ovals on the frame
        if face_in_oval:
            count += 1
            cv2.ellipse(frame, center, axes,
                        0, 0, 360, (0, 255, 0), 2)
            cv2.ellipse(frame, inner_center,
                        inner_axes, 0, 0, 360, (0, 255, 0), 2)
        else:
            cv2.ellipse(frame, center, axes,
                        0, 0, 360, (255, 0, 0), 2)
            cv2.ellipse(frame, inner_center,
                        inner_axes, 0, 0, 360, (255, 0, 0), 2)
        if count == 50:
            width //= 1.5
            height //= 1.5
            width = int(width)
            height = int(height)
            text = "Please zoom out"
            count += 1
        elif count == 100:
            text = "You showed your liveness succesfully"
        coordinates = (25, 25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = cv2.putText(frame, text, coordinates, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', image)
        # Yield the JPEG frame as bytes
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


@app.route('/video_feed')
def video_feed():
    # Return the response with the streaming content
    return Response(capture_frames(oval_center, text),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return '''<!doctype html>
              <html>
                <head>
                  <title>Liveness Check</title>
                </head>
                <body>
                  <h1>Video Streaming</h1>
                  <img src="/video_feed" width="640" height="480">
                </body>
              </html>'''


if __name__ == '__main__':
    app.run()
