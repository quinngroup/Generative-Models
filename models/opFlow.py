import cv2
import numpy as np
import time

cap = cv2.VideoCapture("number.mp4")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
print(hsv[...,0].shape)
hsv[...,1] = 255
fps=10
delay=int(1000/fps)
frame=1
out = cv2.VideoWriter('optical_flow1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (hsv[...,0].shape))
while(ret):
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.8, 6, 15, 10, 5, 1.2, 0)
        print(flow.shape)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)
        k = cv2.waitKey(delay) & 0xff
        if k == 27:
            break
        #elif k == ord('s'):
            #cv2.imwrite('opticalfb.png',frame2)
            #cv2.imwrite('opticalhsv.png',bgr)
        prvs = next
        frame+=1
        # Write the frame into the file 
        out.write(bgr)

        #time.sleep(delay)
    else:
        break
cap.release()
out.release()

cv2.destroyAllWindows()
