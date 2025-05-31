import base64
import json
import os
import queue
import time

from ultralytics import YOLO, YOLOE, YOLOWorld

import cv2
import numpy as np
from dotenv import load_dotenv, find_dotenv
import threading

from Camera.CameraStream import CameraStream
from Logger import Logger
from MQTTClient import mqtt_client

MAX_FRAME_COUNT = 1000000000000000000

# --- Load environment variables ---
load_dotenv(dotenv_path=find_dotenv(".env"), override=False)
local_env = find_dotenv(".env.local")
if local_env:
    load_dotenv(dotenv_path=local_env, override=True)
# --- Initialize YOLO model ---
model = YOLO("./models/yolov11_cozmo.pt")

# Start threaded camera stream
stream = CameraStream(0)

fps_limit = 4  # Or whatever you want
frame_time = 1.0 / fps_limit

def init_broker():
    mqtt_client.set_config(
        broker=os.environ.get("MQTT_BROKER", "localhost"),
        port=int(os.environ.get("MQTT_PORT", 1884)),
        username=os.environ.get("MQTT_USERNAME", 'username'),
        password=os.environ.get("MQTT_PASSWORD", 'password'),
    )
    mqtt_client.start()
    mqtt_client.wait_for_broker_ready(timeout=10)

previous_position = None

def visual_cue_handler():
    while True:
        start_time = time.time()

        # Always grab the latest frame from the thread
        ret, frame = stream.read()
        if not ret or frame is None:
            print("Camera disconnected or no frame.")
            break

        # Run YOLO tracking on the newest frame
        results = model.track(frame, conf=0.65, persist=True, tracker="bytetrack.yaml")

        for r in results:
            detected_objects = []
            for box in r.boxes:
                # use box.id for tracking ID if available to str
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detected_objects.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                })

            Logger.debug(f"[YOLO] Detections: {json.dumps(detected_objects)}")

            # main cozmo : get the detection with the highest confidence
            main_cozmo = None
            if detected_objects:
                main_cozmo = max(detected_objects, key=lambda x: x['confidence'])
                # get center of the bounding box
                x1, y1, x2, y2 = main_cozmo['bbox'].values()
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # set coordinates for the main cozmo relative to the frame 640x480
                current_coordinates = {
                    "x": center_x / 640.0,
                    "y": center_y / 480.0
                }
                # todo check if moving or not, smooth with kalman filter
                # it need to be precise enought so that cozmo is able to know if he is stuck or not
                # <CODE AFTER>

                # <CODE BEFORE>
                Logger.info(f"[YOLO] Coordinates: {json.dumps(current_coordinates)}")
                previous_position = current_coordinates


            # Publish the frame to the MQTT broker
            img_bytes =  cv2.imencode('.jpg', r.plot())[1].tobytes()
            # publish the image to a different topic
            now = time.time().__str__()
            # image is base64 encoded
            object_to_publish = {
                "stamp": now,
                "image": base64.b64encode(img_bytes).decode('utf-8')
            }
            mqtt_client.publish("cozmo/camera_top/annotated", json.dumps(object_to_publish))

        # Time control to limit FPS
        elapsed = time.time() - start_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    init_broker()
    Logger.info("[Main] Starting visual cue service...")
    # start thread to handle visual cues
    threading.Thread(target=visual_cue_handler, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        Logger.info("[Main] Shutting down")
        mqtt_client.stop()
        # end of while loop, stop the stream
        stream.stop()
