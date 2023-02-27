import numpy as np
import cv2
from collections import deque
import gdown


### START ###
in_file = "video_in.mp4"
out_file = "video_out.avi"
OUT_RATIO = 2 # size ratio of display & output video
class_names = ['butterfly']


# trained yolo model
CONFIDENCE_THRESHOLD = 0.2 # param
NMS_THRESHOLD = 0.4 # param
url = "https://drive.google.com/file/d/1oOAePG8qiM0ZL_gBTv20JjvY5Nu94oho/view?usp=sharing"
gdown.download(url, "yolov3.weights", quiet=False, fuzzy=True)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


# initialize tracker
MAX_DIST = 80 # param: take detected obj as an existing obj if min dist from old centroids < MAX_DIST
NUM_FRAME_KEEP = 4 # param: num of centroids kept in tracker (1 : discard all old detected objs)
tracker = {} # {obj_id: (centroid, class_name, score, box, points), ...}
next_id = 0 # tracked obj id


# load video
cap = cv2.VideoCapture(in_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * OUT_RATIO
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * OUT_RATIO
video_writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('X','V','I','D'), fps, (width, height))

i_frame = 0
track_count = 0
try:
    while True:
        # current video time
        minutes = i_frame // fps // 60
        seconds = (i_frame // fps) % 60
        milliseconds = ((i_frame * 1000) // fps) % 1000
        curr_time = (minutes, seconds, milliseconds)

        # read frame
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, None, fx=OUT_RATIO, fy=OUT_RATIO)

        # detect obj by yolo
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)


        ### ------- TRACKING OBJECTS ------- ###
        # track detected objs
        new_tracker = {}
        for (classid, score, box) in zip(classes, scores, boxes):
            class_name = class_names[classid]
            centroid = box[:2] + box[2:] // 2

            # find min dist from old objs
            min_dist = 99999
            min_id = -1
            for old_id, (old_centroid, old_class_name, _, _, _) in tracker.items():
                if old_class_name == class_name: # must be same class
                    dist = np.sqrt( np.sum( (centroid - old_centroid)**2 ) )
                    if dist < min_dist:
                        min_dist = dist
                        min_id = old_id

            # create a new tracking record for a new obj
            if min_dist > MAX_DIST:
                points = deque([tuple(centroid)], maxlen=NUM_FRAME_KEEP) # new record of centroids
                # deque automatically pop oldest one by keeping only "NUM_FRAME_KEEP" frames
                new_tracker[next_id] = (centroid, class_name, curr_time, box, points)
                next_id += 1

            # update tracking record for an old obj detected in current frame
            else:
                points = tracker[min_id][-1] # tracked centroids
                points.append(tuple(centroid))
                last_time = tracker[min_id][2]
                new_tracker[min_id] = (centroid, class_name, last_time, box, points)

        # handle missed objs
        for old_id, (old_centroid, class_name, last_time, _, points) in tracker.items():
            if old_id not in new_tracker:

                # Add "None" for missed obj
                points.append(None)

                # discard obj if num_misses == "NUM_FRAME_KEEP"; keep obj in tracker otherwise
                if points.count(None) < NUM_FRAME_KEEP:
                    new_tracker[old_id] = (old_centroid, class_name, last_time, None, points)
                    
        # update tracker
        tracker = new_tracker
        print(f"Frame {i_frame} : {sorted(tracker)}") # print tracked obj ids
        ### -------------------------------- ###

        
        # draw tracking points
        for _, (_, _, _, box, points) in tracker.items():
            if box is not None:
                points = [pt for pt in points if pt is not None]
                for pt_1, pt_2 in zip(points[:-1], points[1:]):
                    cv2.circle(frame, pt_1, 4, (0,255,255), -1)
                    cv2.line(frame, pt_1, pt_2, (0,255,255), 2)
        
        # draw boxes of objs
        for id, (centroid, class_name, last_time, box, _) in tracker.items():
            if box is not None:
                cv2.circle(frame, centroid, 4, (0,0,255), -1)
                cv2.rectangle(frame, box, (0,255,0), 2)
                label = f"{class_name}({id}) {last_time[0]}:{last_time[1]:02d}.{last_time[2]:03d}"
                cv2.putText(frame, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # display current time & obj count
        label = f"TIME:{minutes}:{seconds:02d}.{milliseconds:03d}  COUNT:{next_id}"
        cv2.putText(frame, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # save video
        video_writer.write(frame)

        # display on screen
        cv2.imshow('FRAME', frame)
        if cv2.waitKey(1) == ord('q'): # press 'q' to exit
            break
        i_frame += 1
        
finally:
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()



