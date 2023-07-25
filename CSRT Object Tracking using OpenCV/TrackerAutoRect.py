import cv2
import sys
import time

tracker_type = "CSRT"
tracker = cv2.TrackerCSRT_create()

center_coordinate = []
quit = 1
second_quit = 1
roi = ()
def draw_rectangle(event,x,y,flags,param):
    global center_coordinate,quit,roi
    if event == cv2.EVENT_LBUTTONDOWN:
        center_coordinate = [(x,y)]
        print((x,y))
        cv2.rectangle(frame,(x-25,y-25),(x+25,y+25),(255,0,0),2)
        roi = (x-25,y-25,x+25,y+25)
        print(roi)
        quit = 0

def mouse_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global second_quit
        second_quit = 0

video = cv2.VideoCapture("../YoloLast/basketball.mp4")

ok, frame = video.read()
print(frame.shape)

if not ok:
    print('Video cannot be read')
    sys.exit()

cv2.namedWindow('Showing')
cv2.setMouseCallback('Showing',draw_rectangle)


while quit:
    ok, frame = video.read()
    if not ok:
        break
    cv2.imshow('Showing', frame)
    key = cv2.waitKey(20)
    if key == 13:
        break

ok = tracker.init(frame, roi)
cv2.setMouseCallback('Showing',mouse_click)
while second_quit:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

 
    timer = cv2.getTickCount()
    ok, roi = tracker.update(frame)
    print(roi)

   
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if ok:
        p1 = (int(roi[2]), int(roi[3]))
        p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

   
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    cv2.imshow('Showing', frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

while True:
    ok, frame = video.read()
    if not ok:
        break
    cv2.imshow('Showing', frame)
    key = cv2.waitKey(20)
    if key == 27:
        break

