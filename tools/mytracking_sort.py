import numpy as np
import time
from collections import defaultdict
import collections

from sort_master.sort import Sort

'''
Sample usage on bottom!
'''

class uniqueId:
    #giving new id base on first in
    def __init__(self):
        self.arrId = {}
        self.counter = 1
        
    def add_get_id(self,currentId):
        """try to get id"""
        try:
            realId = self.arrId[currentId]
        except:
            self.arrId[currentId] = self.counter
            realId = self.counter
            self.counter+= 1
        return realId

class MyTrackingSort:
    def __init__(self, lenRecord=30, distance_th=0.005, time_th=0.5):
        self.n_id = 0
        self.th_dist = distance_th
        self.lenRecord = lenRecord
        self.ts_threshold = time_th
        self.object_track = collections.namedtuple('Track', ['ts', 'box', 'centroid'])
        self.uId = uniqueId()
        self.trackDict = defaultdict(list)
            
    def init_sort(self, max_age=1, min_hits=3):
        self.mot_tracker = Sort(max_age, min_hits)
        
    def update(self, boxes):
        trackers = self.mot_tracker.update(np.array(boxes))
        return trackers
    
    def get_tracker(self, trackers, ts):
        self.trackDictNow = {}
        for track in trackers:
            x1, y1, x2, y2, objectId = track.astype(int)
            bbox = x1, y1, x2, y2
            # mine start here
            centroid = get_center(bbox) 
            object_track_now = self.object_track(ts=ts, box=bbox, centroid=centroid)
            self.trackDictNow[objectId] = object_track_now
            self._add_centroids(self.trackDictNow) 
            self._tracker_logic()
        
    
    def _add_centroids(self, trackDictNow):
        # 1. Get all id
        id_registered = self.trackDict.keys()
        # 2. Add object_track_now to database(trackDict) 
        for objectId in trackDictNow:
            # 2.a not registered
            if objectId not in id_registered:
                # 2x add for tracking requirement
                self.trackDict[objectId].append(trackDictNow[objectId])
                self.trackDict[objectId].append(trackDictNow[objectId])
                
            else:
            # 2.b registered -> eliminate / delete 
                self.trackDict[objectId].append(trackDictNow[objectId])
    
    def _tracker_logic(self):
        '''
        Delete, if:
            -> no longer move (th_dist) in (ts_threshold) seconds 
        '''
        
        arr_delete = []
        for objectId in self.trackDict:
            # --- maintain len trackdict[objectid]
            if len(self.trackDict[objectId]) > self.lenRecord:
                self.trackDict[objectId] = self.trackDict[objectId][1:]
            
            # --- Main Logic
            # config
            ts_now = time.time()
            track_now = self.trackDict[objectId][-1]                                # ts, box, centroid
            ts_last_det = track_now.ts
            centroid_now = track_now.centroid
            tracks_prev = np.array(self.trackDict[objectId])
            centroids_prev = tracks_prev[:, 2]
            tss_prev = tracks_prev[:, 0]
            # -- 1) Logic not move
            not_move = self._is_not_move(ts_now, tss_prev, centroid_now, centroids_prev)
            # -- 2) Logic not detected in ts_th second
            is_timeout = self._is_timeout(ts_now, ts_last_det)
            if not_move or is_timeout:
                arr_delete.append(objectId)
        # -- 2) deleting Id
        self._delete_ids(arr_delete)
            
    #static
    def _is_timeout(self, ts_now, ts_last_det):
        if ts_now - ts_last_det > self.ts_threshold:
            return True
        else:
            return False
    
    def _is_not_move(self, ts_now, tss_prev, centroid_now, centroids_prev):
        track_timeout = np.where(ts_now - tss_prev > self.ts_threshold)[0]
        if len(track_timeout) > 0:
            last_track_timeout = track_timeout[-1]               # get last
            centroid_timeout = centroids_prev[last_track_timeout]
            dist = np.linalg.norm(abs(centroid_now - centroid_timeout))
            if dist < self.th_dist :
                return True
        return False
            
    def _delete_ids(self, arr_delete):
        for i in arr_delete:
           print("deleting id ->",i)
           del self.trackDict[i]
        
    
    def get_array(self):
        return self.trackDict
    
  

### static
def get_center(coord):
    x = (coord[0]+coord[2])*0.5
    y = (coord[1]+coord[3])*0.5
    return np.array((x,y))

def to_width_height_box(boxes):
    array = []
    for box in boxes:
        w = abs(box[0] - box[2])
        h = abs(box[1] - box[3])
        array.append([box[0], box[1], w, h])
    return array
    
    
'''
## 1) import
from tools.mytracking_sort import MyTrackingSort

## 2) Init
mt = MyTrackingSort(lenRecord=30, distance_th=0.005, time_th=1)
mt.init_sort()

## 3) while loop
ts = time.time()
tracker = mt.update(boxes)
mt.get_tracker(tracker, ts)
trackDict = mt.get_array()

## 4) visualize
for objectId in list(trackDict):
try:
    bbox = np.array(trackDict[objectId])[-1,1]
except:
    bbox = np.array(trackDict[objectId])[0,1]
# vis tracker bbox
cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
# vis id
cv2.putText(frame, str(objectId),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 100, (0,255,0),2)
arrCentroid = np.array(trackDict[objectId])[:,2]
# vis line tracker obj
for i,v in enumerate(arrCentroid):
    if i>1:
        cv2.line(frame, (int(arrCentroid[i][0]), int(arrCentroid[i][1])), (int(arrCentroid[i-1][0]),int(arrCentroid[i-1][1])), (0, 255, 0), 2)


'''