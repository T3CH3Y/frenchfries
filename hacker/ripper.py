import cv2
import mediapipe as mp
import time
import math
 
 
class poseDetector():
 
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
 
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode = self.mode, smooth_landmarks=self.upBody, smooth_segmentation=self.smooth,min_detection_confidence=self.detectionCon, min_tracking_confidence =self.trackCon)
 
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
 
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                h, w, c = img.shape
                # print this shit
                cx, cy, cz = int(lm.x), int(lm.y), int(lm.z)
                self.lmList.append([id, cx, cy, cz])
        return self.lmList
 
 
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    set = open
    with open("dataman.csv", "a") as set:
        while (True):
            time.sleep(0.2)
            success, img = cap.read()
            img = detector.findPose(img)
            coords = detector.findPosition(img, draw = False)
            
            
 
 
if __name__ == "__main__":
    main()