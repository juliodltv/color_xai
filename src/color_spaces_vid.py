import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
    
import cv2, numpy as np

class ColorCalibration:
    def __init__(self, video_input, color_space):
        self.video_input = video_input  
        self.color_space = color_space
        self.labels = ['CH1L', 'CH2L', 'CH3L', 'CH1H', 'CH2H', 'CH3H']
        self.values = [255, 255, 255, 255, 255, 255]
        self.default = [0, 0, 0, 255, 255, 255]
        
        cv2.namedWindow('Color')
        for l, v, d in zip(self.labels, self.values, self.default): cv2.createTrackbar(l, 'Color', d, v, self.nothing)
        
        self.camera = cv2.VideoCapture(video_input)
        self.image_callback()
        
    def nothing(self, value): pass
    
    def image_callback(self):
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret: return
            frame = frame[0:800, :, :]
            
            img = cv2.imread('images/color_wheel.png')
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Color', img)
            
            CH1L, CH2L, CH3L, CH1H, CH2H, CH3H = [cv2.getTrackbarPos(l, 'Color') for l in self.labels]

            # cv2.imshow('Color', img)
            cv2.imshow('Video', frame)
            
            hls = cv2.cvtColor(frame, self.color_space)
            
            low, high = np.asarray([CH1L, CH2L, CH3L], np.uint8), np.asarray([CH1H, CH2H, CH3H], np.uint8)
            mask = cv2.inRange(hls, low, high)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Mask', mask)
            if cv2.waitKey(60) == 27: self.stop()

    def stop(self):
        self.camera.release()
        cv2.destroyAllWindows()

def main():
    video_path = "real_data/video.mp4"
    color_calibration = ColorCalibration(video_path, color_space=cv2.COLOR_BGR2HLS_FULL)
    color_calibration.stop()
    
if __name__ == "__main__":
    main()