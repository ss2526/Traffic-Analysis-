import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from sort import Sort
import math
import time

# Define video path and initialize video capture
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt')

# Read class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Define road zones
road_zoneA = np.array([[636, 451], [962, 456], [948, 823], [412, 798], [636, 454]], np.int32)
road_zoneB = np.array([[966, 458], [1272, 471], [1443, 750], [969, 829], [969, 464]], np.int32)
road_zoneC = np.array([[1275, 475], [1517, 458], [1800, 668], [1453, 748], [1276, 484]], np.int32)




# Define tracking
tracker = Sort()
zoneAcounter = []
zoneBcounter = []
zoneCcounter = []

# Dictionary to store vehicle positions and timestamps
vehicle_data = {}

# Set meters per pixel based on real-world scale (calibrate this value based on actual measurements)
meters_per_pixel = 0.06  # Adjust this value based on real-world dimensions

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1920, 1080))
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds

    results = model(frame)
    current_detections = np.empty([0, 5])

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if class_detect in ['car', 'truck', 'bus'] and conf > 60:
                detections = np.array([x1, y1, x2, y2, conf])
                current_detections = np.vstack([current_detections, detections])

    # Draw road zones and lines
    # cv2.polylines(frame, [road_zoneA], isClosed=False, color=(0, 0, 255), thickness=8)
    # cv2.polylines(frame, [road_zoneB], isClosed=False, color=(0, 255, 255), thickness=8)
    # cv2.polylines(frame, [road_zoneC], isClosed=False, color=(255, 0, 0), thickness=8)
    # cv2.line(frame, line1_start, line1_end, (0, 255, 0), 2)
    # cv2.line(frame, line2_start, line2_end, (255, 0, 0), 2)

    # Update tracking
    track_results = tracker.update(current_detections)
    for result in track_results:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Check if the vehicle has been tracked before
        if id in vehicle_data:
            prev_cx, prev_cy, prev_time = vehicle_data[id]

            # Calculate distance in pixels between the current and previous position
            distance_pixels = np.linalg.norm(np.array((cx, cy)) - np.array((prev_cx, prev_cy)))
            time_elapsed = current_time - prev_time

            # Calculate speed if time has elapsed
            if time_elapsed > 0:
                distance_meters = distance_pixels * meters_per_pixel
                speed_mps = distance_meters / time_elapsed  # speed in meters per second
                speed_kmph = speed_mps * 3.6  # Convert m/s to km/h
                cv2.putText(frame, f'Speed: {speed_kmph:.2f} km/h', (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Update the vehicle data with the new position and timestamp
            vehicle_data[id] = (cx, cy, current_time)
        else:
            # Initialize vehicle data
            vehicle_data[id] = (cx, cy, current_time)

        # Counting logic
        if road_zoneA[0][0] < cx < road_zoneA[1][0] and road_zoneA[0][1] - 20 < cy < road_zoneA[1][1] + 20:
            if id not in zoneAcounter:
                zoneAcounter.append(id)

        if road_zoneB[0][0] < cx < road_zoneB[1][0] and road_zoneB[0][1] - 20 < cy < road_zoneB[1][1] + 20:
            if id not in zoneBcounter:
                zoneBcounter.append(id)

        if road_zoneC[0][0] < cx < road_zoneC[1][0] and road_zoneC[0][1] - 20 < cy < road_zoneC[1][1] + 20:
            if id not in zoneCcounter:
                zoneCcounter.append(id)

    # Display vehicle counts
    cv2.circle(frame, (970, 90), 15, (0, 0, 255), -1)
    cv2.circle(frame, (970, 130), 15, (0, 255, 255), -1)
    cv2.circle(frame, (970, 170), 15, (255, 0, 0), -1)
    cvzone.putTextRect(frame, f'LANE A Vehicles = {len(zoneAcounter)}', [1000, 99], thickness=4, scale=2.3, border=2)
    cvzone.putTextRect(frame, f'LANE B Vehicles = {len(zoneBcounter)}', [1000, 140], thickness=4, scale=2.3, border=2)
    cvzone.putTextRect(frame, f'LANE C Vehicles = {len(zoneCcounter)}', [1000, 180], thickness=4, scale=2.3, border=2)

    # Show the frame
    # Resize frame to reduce computation load
    small_frame = cv2.resize(frame, (640, 360))

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == 27:  # Exit on 'Esc'
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
