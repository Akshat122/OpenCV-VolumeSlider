import cv2
import mediapipe as mp 

# Load the config file
import config

class HandDetection:
    def __init__(self):
        self.handsModule = mp.solutions.hands
        self.hands = self.handsModule.Hands(config.static_img_mode,config.max_num_hands,config.min_detection_confidence ,config.min_tracking_confidence)
        self.drawFun = mp.solutions.drawing_utils

    def detectHands(self,image):
        imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # Converting the BGR camera feed into RGB colour 
        self.handTracking = self.hands.process(imgRGB) # Look for hands in the image
        if self.handTracking.multi_hand_landmarks: # For all the hands points detected by the hands module
            for handLandMarks in self.handTracking.multi_hand_landmarks:
                if config.drawHandLandmarks:
                    self.drawFun.draw_landmarks(image,handLandMarks, self.handsModule.HAND_CONNECTIONS)
        return image
    
    def displayLandmarks(self,image, hand=0,draw=True):
        landMarksList = []
        if self.handTracking.multi_hand_landmarks: # For all the hands points detected by the hands module
            hand = self.handTracking.multi_hand_landmarks[hand]
            height, width, center = image.shape
            for id, landMarks in enumerate(hand.landmark):
                x , y = int(landMarks.x * width) , int(landMarks.y * height)
                landMarksList.append([id,x,y])
        
        return landMarksList



def main():
    capture = cv2.VideoCapture(config.videoCapCard) 
    capture.set(3,config.widthCam)
    capture.set(4,config.heightCam)
    detectorObj = HandDetection()
    while True:
        success, img = capture.read()
        img = detectorObj.detectHands(img)
        lmList = detectorObj.displayLandmarks(img)

        if len(lmList)>0:
            print(lmList[4])

        cv2.imshow("Image",img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
    