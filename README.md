# Cozmo Visual Cue Service (model included)

**A real-time object detection and tracking service for Anki Cozmo using YOLOv11 fine-tuned model and MQTT.**
Detects and tracks your robot with your webcam, sends live annotated images, and publishes robot activity events to MQTT for seamless robotics integration.

---

![annotated image](https://github.com/arkdevuk/cozmo_spatial_cue/raw/master/Doc/images/cozmo_found.jpeg)

---

## Features

* **Real-time webcam object detection & tracking** using \[Ultralytics YOLOv11]
* **Custom YOLOv11 model included** fine-tuned for Cozmo robot detection from top-down webcam view
* **MQTT event publishing**:
  * `robot_found` / `robot_lost` when Cozmo appears/disappears
  * `motion` events when the robot is moving or stuck
  * Live annotated camera images
* **Automatic hardware acceleration** (CUDA if available)
* **Configurable** via `.env` or `.env.local`

---

## How it Works

* The service runs a YOLOv detection model on webcam frames.
* When Cozmo is detected, it broadcasts:
  * Annotated image frames (JPG, base64-encoded) on MQTT topic `cozmo/camera_top/annotated`
  * Robot presence events on `cozmo/camera_top/events`
  * Motion status events on `cozmo/camera_top/motion`
* If Cozmo disappears for a short period (configurable), a `robot_lost` event is sent.
* If Cozmo is detected again, a `robot_found` event is sent.
* Motion status (`moving` or not) is determined based on the center of the bounding box, filtered by simple logic to avoid noise (Kalman was too slow).

---

![logs](https://github.com/arkdevuk/cozmo_spatial_cue/raw/master/Doc/images/logs.png)

---

## MQTT Topics

* `cozmo/camera_top/annotated` — **Live annotated image (JPG base64, with timestamp)**
* `cozmo/camera_top/events` — **Robot presence**

  * `{ "type": "robot_found" }`
  * `{ "type": "robot_lost" }`
* `cozmo/camera_top/motion` — **Motion event**

  * `{ "state": true|false, "stamp": "timestamp" }`

---

## Example `.env`:

```
MQTT_BROKER=localhost
MQTT_PORT=1884
MQTT_USERNAME=username
MQTT_PASSWORD=password
```

---

## Code Overview

* `main.py` — Main script, event loop, detection & MQTT publishing
* `Camera/CameraStream.py` — Threaded webcam capture
* `Logger.py` — Logging abstraction
* `MQTTClient/MQTTClient.py` — MQTT wrapper for easy publishing

---

## Extending/Integrating

* Use the MQTT topics in your own robot control scripts or dashboards.
* Tune detection, presence, and motion parameters in `main.py`.
* Swap the model or retrain on your custom dataset for more objects or different robots.

---

## License

[MIT License](LICENSE)

---

## Credits

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [Anki Cozmo SDK](https://github.com/anki/cozmo-python-sdk)
* [PyCozmo](https://github.com/zayfod/pycozmo)
* MQTT with [paho-mqtt](https://www.eclipse.org/paho/)

---

***Pull requests welcome!** For questions or ideas, open an issue or discussion.*