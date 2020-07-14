
import os
import cv2
import numpy as np
import time

from tools.mytracking_sort import MyTrackingSort

if __name__ == '__main__':
    ## -----------------------------------------
    ## Init
    ## -----------------------------------------
    # a. Model
    mt = MyTrackingSort(lenRecord=30, distance_th=0.005, time_th=0.5)
    mt.init_sort(max_age=1, min_hits=2)
    
    # Video
    cap = cv2.VideoCapture(vidSrc)
      
    while cap.isOpened():
        ret, frame = cap.read()
        ts = time.time()
        ## ---
        ## 1 Detection 
        boxes = []
        boxes, scores = detection.predict(frame)
        ## ---
        ## 2 Tracking
        ts = time.time()
        tracker = mt.update(boxes)
        mt.get_tracker(tracker, ts)
        trackDict = mt.get_array()
                      
        ## -----------------------------------------
        ## Visualization
        ## -----------------------------------------
        # Visualization Detection
        for box in boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(255,255,0), 2)
        # Visualization Tracking
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
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
