from collections import deque
import cv2
import queue

class CvDrawBuffer:
    def __init__(self, window_name = 'Buffer', resolution=(320,320)):
        self.buffer = deque(maxlen=1) # Was maxlen=2
        self.first_draw = True
        self.window_name = window_name
        self.resolution = resolution

    def PushFrame(self, frame):
        self.buffer.append(frame)

    def Draw(self):
        if len(self.buffer) > 0:
            if self.first_draw == True:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, self.resolution[0], self.resolution[1])
                self.first_draw = False

            cv2.imshow(self.window_name, self.buffer[0])