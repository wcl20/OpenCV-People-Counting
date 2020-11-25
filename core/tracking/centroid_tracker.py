import numpy as np
from collections import OrderedDict
from scipy.spatial import distance

class CentroidTracker:

    def __init__(self, max_disappear_frames=50, max_distance=50):
        self.next_id = 0
        self.objects = OrderedDict()
        # Maximum frames to deregister disappeared objects
        self.max_disappear_frames = max_disappear_frames
        self.disappeared = OrderedDict()
        # Maximum distance between centroids to associate objects
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, id):
        del self.objects[id]
        del self.disappeared[id]

    def update(self, bboxs):
        if len(bboxs) == 0:
            for id in list(self.disappeared.keys()):
                self.disappeared[id] += 1
                if self.disappeared[id] > self.max_disappear_frames:
                    self.deregister(id)
            return self.objects
        # Compute centroids for each bbox
        input_centroids = np.zeros((len(bboxs), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(bboxs):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids[i] = (cx, cy)
        # No currently tracking objects
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Currently tracking objects
            tracking_ids = list(self.objects.keys())
            tracking_centroids = list(self.objects.values())

            D = distance.cdist(np.array(tracking_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()
            for row, col in zip(rows, cols):
                # Ignore examined rows and columns
                if row in used_rows or col in used_cols:
                    continue
                # Ignore if distance between centroids exceed threshold
                if D[row, col] > self.max_distance:
                    continue
                # Assign centroid to object
                self.objects[tracking_ids[row]] = input_centroids[col]
                self.disappeared[tracking_ids[row]] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Find unused ids and centroids
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            # Some tracking objects disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    self.disappeared[tracking_ids[row]] += 1
                    if self.disappeared[tracking_ids[row]] > self.max_disappear_frames:
                        self.deregister(tracking_ids[row])
            # Some new centroids appeared
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        return self.objects
