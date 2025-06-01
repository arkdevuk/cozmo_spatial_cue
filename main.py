import base64
import json
import os
import time

import torch
from ultralytics import YOLO

import cv2
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
Logger.info(f"[YOLO] Model loaded on {device}")

# Start threaded camera stream
stream = CameraStream(0)

fps_limit = 12
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

def publish_annotated_image(img, topic="cozmo/camera_top/annotated"):
    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
    now = str(time.time())
    object_to_publish = {
        "stamp": now,
        "image": base64.b64encode(img_bytes).decode('utf-8')
    }
    mqtt_client.publish(topic, json.dumps(object_to_publish))

def publish_motion_event(is_moving, topic="cozmo/camera_top/motion"):
    event = {
        "state": is_moving,
        "stamp": str(time.time()),
    }
    mqtt_client.publish(topic, json.dumps(event))

def publish_robot_event(event_type, topic="cozmo/camera_top/events"):
    event = {
        "type": event_type,
        "stamp": str(time.time()),
    }
    mqtt_client.publish(topic, json.dumps(event))

previous_position = None
previous_state = None
state_history = []

cozmo_found = False
missing_counter = 0

def visual_cue_handler():
    global previous_position, previous_state, state_history
    global cozmo_found, missing_counter
    while True:
        start_time = time.time()

        ret, frame = stream.read()
        if not ret or frame is None:
            Logger.error("[YOLO] Camera disconnected or no frame.")
            break

        results = model.track(frame, conf=0.65, persist=True,
                              verbose=False,
                              tracker="bytetrack.yaml")
        main_cozmo_detected = False

        for r in results:
            detected_objects = []
            for box in r.boxes:
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

            if detected_objects:
                # --- COZMO DETECTION FOUND ---
                main_cozmo = max(detected_objects, key=lambda x: x['confidence'])
                main_cozmo_detected = True

                x1 = main_cozmo["bbox"]["x1"]
                y1 = main_cozmo["bbox"]["y1"]
                x2 = main_cozmo["bbox"]["x2"]
                y2 = main_cozmo["bbox"]["y2"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_coordinates = {"x": center_x, "y": center_y}

                # Movement check (threshold=2)
                move_threshold = 2
                if previous_position is not None:
                    dx = current_coordinates["x"] - previous_position["x"]
                    dy = current_coordinates["y"] - previous_position["y"]
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    is_moving = dist > move_threshold
                else:
                    is_moving = True  # First frame, assume moving

                previous_position = current_coordinates

                # State smoothing: require 2 consecutive frames to trigger state change
                state_history.append(is_moving)
                if len(state_history) > 2:
                    state_history.pop(0)

                # Only trigger state change if last 2 frames agree and state is different
                if previous_state is not is_moving:
                    Logger.info(f"[YOLO] Coordinates: {json.dumps(current_coordinates)}, Moving: {is_moving}")
                    publish_motion_event(is_moving)
                    previous_state = is_moving

            # Publish the annotated image using helper
            publish_annotated_image(r.plot())

        # --- COZMO FOUND/LOST LOGIC ---
        if main_cozmo_detected:
            if not cozmo_found:
                cozmo_found = True
                missing_counter = 0
                publish_robot_event("robot_found")
                Logger.info("[YOLO] Robot found, published robot_found event")
            else:
                missing_counter = 0  # reset
        else:
            if cozmo_found:
                missing_counter += 1
                if missing_counter >= fps_limit:
                    cozmo_found = False
                    publish_robot_event("robot_lost")
                    Logger.info("[YOLO] Robot lost, published robot_lost event")
            else:
                missing_counter = 0  # stays zero

        elapsed = time.time() - start_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

if __name__ == '__main__':
    init_broker()
    Logger.info("[Main] Starting visual cue service...")
    threading.Thread(target=visual_cue_handler, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        Logger.info("[Main] Shutting down")
        mqtt_client.stop()
        stream.stop()
