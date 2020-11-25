class TrackableObject:

    def __init__(self, id, centroid):
        self.id = id
        self.centroids = [centroid]
        self.start = centroid
        self.counted = False
