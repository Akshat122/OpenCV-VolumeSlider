import cv2
import numpy as np
import config
import HandTracking
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setting up display paramaters
capture = cv2.VideoCapture(config.videoCapCard)
capture.set(3, config.widthCam)
capture.set(4, config.heightCam)

# Load the HandTracking module
handDetector = HandTracking.HandDetection()

# Volume Slider
slider = 194
sliderWidth = 20

# Load pycaw to control the volume of the system
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


while True:
    success, image = capture.read()
    image = handDetector.detectHands(image)
    lmList = handDetector.displayLandmarks(image)

    # Create Volume slider
    cv2.rectangle(image, (184, 5), (900, 101), (0, 255, 0), 3)
    cv2.line(image, (204, 48), (880, 48), (255, 0, 0), 2)
    cv2.rectangle(image, (slider, 20), (slider +
                  sliderWidth, 76), (0, 0, 255), 2)

    # Check if hands are detected
    if len(lmList) > 0:
        dist = math.hypot(lmList[config.trackingPointIndex2][1] - lmList[config.trackingPointIndex][1], lmList[config.trackingPointIndex2]
                          [2] - lmList[config.trackingPointIndex][2])  # Checking the distance between two tracking points
        # Check if the distance is small and points are within the volume rectangle range
        if(int(dist) < 31 and (lmList[config.trackingPointIndex][1] > 203 and lmList[config.trackingPointIndex][1] < 870)):
            slider = int(lmList[config.trackingPointIndex]
                         [1])  # Update the slider
            # Convert slider value to system volume range
            vol = np.interp(slider, (203, 870), (-65, 0))
            volume.SetMasterVolumeLevel(vol, None)  # Set system volume

    cv2.imshow("Image", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
