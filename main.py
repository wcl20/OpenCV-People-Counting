import cv2
import dlib
import imutils
import numpy as np
import time
from config import config
from core.tracking import CentroidTracker
from core.tracking import TrackableObject
from imutils.video import VideoStream

def main():

    # Load mobilenet
    prototxt = "MobileNetSSD_deploy.prototxt"
    model = "MobileNetSSD_deploy.caffemodel"
    # Classes of mobilenet SSD
    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Camera
    stream = VideoStream(src=0).start()
    time.sleep(2)

    # Centroid tracker
    centroid_tracker = CentroidTracker()
    # dlib trackers
    trackers = []
    # Trackable objects
    trackable_objects = {}

    total_left = 0
    total_right = 0
    total_frames = 0

    while True:

        frame = stream.read()
        frame = imutils.resize(frame, width=500)
        height, width = frame.shape[:2]

        # Convert color scheme for dlib tracker
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxs = []

        # Run detection model every 'SKIP FRAMES'
        if total_frames % config.SKIP_FRAMES == 0:
            # Reset all trackers
            trackers = []
            # Run detection model
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
            net.setInput(blob)
            detections = net.forward()[0, 0]
            for detection in detections:
                # Filter detections with low confidence
                confidence = detection[2]
                if confidence > config.CONF_THRESH:
                    # Filter detections that are not human
                    label = classes[int(detection[1])]
                    if label == "person":
                        bbox = detection[3:7] * np.array([width, height, width, height])
                        bbox = bbox.astype("int")
                        bboxs.append(bbox)
                        # Setup tracker
                        tracker = dlib.correlation_tracker()
                        x1, y1, x2, y2 = bbox
                        tracker.start_track(frame_rgb, dlib.rectangle(x1, y1, x2, y2))
                        trackers.append(tracker)
        else:
            for tracker in trackers:
                tracker.update(frame_rgb)
                position = tracker.get_position()
                bbox = np.array([position.left(), position.top(), position.right(), position.bottom()])
                bbox = bbox.astype("int")
                bboxs.append(bbox)

        # Update centroid tracker
        objects = centroid_tracker.update(bboxs)
        for id, centroid in objects.items():
            # Get trackable object with id
            object = trackable_objects.get(id, None)
            if object is None:
                object = TrackableObject(id, centroid)
            else:
                # Get previous x coordinates of object
                x_coords = [c[0] for c in object.centroids]
                # Get traveling direction of object
                direction = centroid[0] - np.mean(x_coords)
                object.centroids.append(centroid)
                # Count object if not yet counted
                if not object.counted:
                    # Object traveling to the left
                    if direction < -10 and centroid[0] < width // 2 and object.start[0] > width // 2:
                        total_left += 1
                        object.counted = True
                    # Object traveling to the right
                    elif direction > 10 and centroid[0] > width // 2 and object.start[0] < width // 2:
                        total_right += 1
                        object.counted = True
            trackable_objects[id] = object

            # Draw
            cx, cy = centroid
            cv2.putText(frame, f"ID {id}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Draw
        for x1, y1, x2, y2 in bboxs:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Left: {total_left}", (width - 100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, f"Right: {total_right}", (width - 100, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 255), 2)

        # Show frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        total_frames += 1

    cv2.destroyAllWindows()
    stream.stop()




if __name__ == '__main__':
    main()
