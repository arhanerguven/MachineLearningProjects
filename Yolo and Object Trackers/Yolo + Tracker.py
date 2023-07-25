
import numpy as np
import cv2
import time
import sys



video = cv2.VideoCapture('top.mp4')
h, w = None, None

with open('coco.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
layers_names_all = network.getLayerNames()

layers_names_output = \
    [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

probability_minimum = 0.3
threshold = 0.3

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

f = 0

t = 0
track = False

while not track:
    
    ret, frame = video.read()
    if not ret:
        break
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (160, 160),
                                 swapRB=True, crop=False)

    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Increasing counters for frames and total time
    f += 1
    t += end - start

    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))


    bounding_boxes = []
    confidences = []
    classIDs = []

    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:

                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classIDs.append(class_current)
                
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)


  
    if len(results) > 0:
        for i in results.flatten():

            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[classIDs[i]].tolist()

            current_class = labels[int(classIDs[i])]

            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            text_box_current = '{}'.format(labels[int(classIDs[i])])

     
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
            print((x_min, y_min, x_min + box_width, y_min + box_height))

    coordinates = [-1, -1]


    def mouse_click(event, x, y, flags, param):
        global track
        if event == cv2.EVENT_LBUTTONDBLCLK:
            track = True



    cv2.namedWindow('Showing', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Showing', mouse_click)
    cv2.imshow('Showing', frame)
    if cv2.waitKey(1) == ord('q'):
        break


tracker_type = "CSRT"
tracker = cv2.TrackerCSRT_create()

center_coordinate = []
roi = (x_min, y_min, box_width,box_height)


ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

ok = tracker.init(frame, roi)

while track:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break


    timer = cv2.getTickCount()


    ok, roi = tracker.update(frame)
    print(roi)


    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if ok:
        p1 = (int(roi[0]), int(roi[1]))
        p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
     
        cv2.putText(frame, "Tracking failure ", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    cv2.putText(frame, text_box_current, (roi[0], roi[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    cv2.imshow('Showing', frame)

    k = cv2.waitKey(1) 
    if k == ord('q'):
        break

video.release()


