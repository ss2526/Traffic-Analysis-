import cv2
import numpy as np

# Global variables
polygon_points = []
polygons = []

# Read your video file
video_path = r'test.mp4'
cap = cv2.VideoCapture(video_path)

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:  
        polygon_points.append((x, y))
        print(f"Point Added: (X: {x}, Y: {y})")

# Capture the first frame
ret, frame = cap.read()
frame = cv2.resize(frame, (1920, 1080))

# Create a window and set the mouse callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)

while True:
    # Create a copy of the frame to draw on
    frame_copy = frame.copy()

    # Draw all closed polygons
    for poly in polygons:
        cv2.polylines(frame_copy, [np.array(poly)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw the current open polygon
    if len(polygon_points) > 1:
        cv2.polylines(frame_copy, [np.array(polygon_points)], isClosed=False, color=(0, 0, 255), thickness=2)

    # Display the frame copy
    cv2.imshow('Frame', frame_copy)

    # Wait for user input
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == 13:  # Press 'Enter' to close the current polygon
        if len(polygon_points) > 2:  # A polygon requires at least 3 points
            polygons.append(polygon_points)
            polygon_points = []  # Clear current points to start a new polygon

cv2.destroyAllWindows()
cap.release()

# Print all polygons' points
print("Polygons Points:")
for idx, poly in enumerate(polygons):
    print(f"Polygon {idx + 1}: {poly}")



# lines for speed 

