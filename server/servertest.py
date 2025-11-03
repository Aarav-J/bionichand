from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from landmarks import image_processing
import cv2
import time



class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret: 
            return None
        return frame

    def __del__(self):
        self.cap.release()
app = FastAPI()
camera = Camera() # Initialize your Camera class instance

def generate_frames():
    target_fps = 30.0
    target_interval = 1.0/target_fps
    last_time = time.perf_counter() 
    while True: 
        start_time = time.perf_counter()
        frame = camera.get_frame() 
        if frame is None: 
            break 
        image_out = image_processing(frame)
        ret, buffer = cv2.imencode('.jpg', image_out)
        if not ret: 
            elapsed = time.perf_counter() - start_time
            to_sleep = max(0.0, target_interval - elapsed)
            time.sleep(to_sleep)
            continue
        jpg_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        elapsed = time.perf_counter() - start_time
        to_sleep = max(0.0, target_interval - elapsed)
        if to_sleep > 0:
            time.sleep(to_sleep)
    

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
