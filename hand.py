import cv2
import mediapipe as mp 
import time #to calculate frame per second 

#to open web camera
cap = cv2.VideoCapture(0)           #we pass 0 as port number is 0 

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


#calculating frame per second
prevTime = 0
currTime = 0

while True:
     ret,img = cap.read()
     #to convert images from bgr to rgb
     imgBGR = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
     results = hands.process(imgBGR)
     #checking whether multiple hands were extracted or not
     print(results.multi_hand_landmarks) 

     #drawing hand landmarks on our hand
     if results.multi_hand_landmarks :
         for handLandmarks in results.multi_hand_landmarks:   #calculating id's (there are 21 id's in total)
                for id,lm in enumerate(handLandmarks.landmark):
                    #print(id,lm)

                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h) 
                    print(id,cx,cy)

                    if id==4:
                        #drawing the circle on the landmarks
                        cv2.circle(img,(cx,cy),20,(255,0,255),cv2.FILLED)

                mpDraw.draw_landmarks(img,handLandmarks,mpHands.HAND_CONNECTIONS)      #connecting the landmarks on our hands          

     #calculating fps - fps defines how fast our object detection model processes our video and generates the desired output.
     currTime = time.time()
     fps = 1/(currTime - prevTime) #1 for per second  
     prevTime = currTime
     cv2.putText(img , str(int(fps)) ,(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


     cv2.imshow('Hand Gestures',img)
     if cv2.waitKey(1)==13 :          #if we press enter the web camera stops
         break
cv2.destroyAllWindows()

