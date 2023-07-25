import cv2
import sys
import time

tracker_type = "CSRT"
tracker = cv2.TrackerCSRT_create()

center_coordinate = []
quit = 1
second_quit = 1
roi = (287, 23, 86, 320)
def draw_rectangle(event,x,y,flags,param):
    global frame,quit,roi
    if event == cv2.EVENT_LBUTTONDBLCLK:
       roi = cv2.selectROI('Showing',frame,False)
       print(roi)
       quit = 0

def mouse_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global second_quit
        second_quit = 0

video = cv2.VideoCapture("football2.mp4")

ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

cv2.namedWindow('Showing')
cv2.setMouseCallback('Showing',draw_rectangle)


while quit:
    ok, frame = video.read()
    if not ok:
        break
    cv2.imshow('Showing', frame)
    key = cv2.waitKey(20)
    if key == 27:
        break

ok = tracker.init(frame, roi)
cv2.setMouseCallback('Showing',mouse_click)

while second_quit:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, roi = tracker.update(frame)
    print(roi)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if ok:
        p1 = (int(roi[0]), int(roi[1]))
        p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow('Showing', frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

while True:
    ok, frame = video.read()
    if not ok:
        break
    cv2.imshow('Showing', frame)
    key = cv2.waitKey(20)
    if key == 27:
        break
'''
    if ok:
        width = int(abs(roi[2]-roi[0])/2)
        length = int(abs(roi[3]-roi[1])/2)
        constant = 50
        if roi[2] > roi[0]:
            centerx = roi[0] + width
        else:
            centerx = roi[2] + width

        if roi[3] > roi[1]:
            centery = roi[1] + length
        else:
            centery = roi[3] + length

        p1 = (int(centerx - width),int(centery-length))
        p2 = (int(centerx + width),int(centery +length))
'''
