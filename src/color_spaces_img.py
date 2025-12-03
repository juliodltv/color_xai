import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
    
import cv2, numpy as np

class ColorCalibration:
    def __init__(self, color_space):
        self.color_space = color_space
        self.labels = ['CH1L', 'CH2L', 'CH3L', 'CH1H', 'CH2H', 'CH3H']
        self.values = [255, 255, 255, 255, 255, 255]
        self.default = [0, 0, 0, 255, 255, 255]
        
        self.img1 = cv2.imread("simulated_data/image_10.png")
        self.img2 = cv2.imread("simulated_data/image_40.png")
        self.img3 = cv2.imread("simulated_data/image_255.png")
        
        self.img1 = cv2.resize(self.img1, (0, 0), fx=0.5, fy=0.5)
        self.img2 = cv2.resize(self.img2, (0, 0), fx=0.5, fy=0.5)
        self.img3 = cv2.resize(self.img3, (0, 0), fx=0.5, fy=0.5)
        
        cv2.namedWindow('Color')
        for l, v, d in zip(self.labels, self.values, self.default): cv2.createTrackbar(l, 'Color', d, v, self.nothing)
        
        # self.image_callback()
        
    def nothing(self, value): pass
    
    def image_callback(self):
        while True:
            
            img = cv2.imread('images/color_wheel.png')
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Color', img)
            
            CH1L, CH2L, CH3L, CH1H, CH2H, CH3H = [cv2.getTrackbarPos(l, 'Color') for l in self.labels]

            # cv2.imshow('img1', self.img1)
            # cv2.imshow('img2', self.img2)
            # cv2.imshow('img3', self.img3)
            
            row2 = np.hstack((self.img1, self.img2, self.img3))
            
            hls1 = cv2.cvtColor(self.img1, self.color_space)
            hls2 = cv2.cvtColor(self.img2, self.color_space)
            hls3 = cv2.cvtColor(self.img3, self.color_space)
            
            low, high = np.asarray([CH1L, CH2L, CH3L], np.uint8), np.asarray([CH1H, CH2H, CH3H], np.uint8)
            mask1 = cv2.inRange(hls1, low, high)
            mask2 = cv2.inRange(hls2, low, high)
            mask3 = cv2.inRange(hls3, low, high)
            
            self.mask1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
            self.mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
            self.mask3 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)
            
            row1 = np.hstack((self.mask1, self.mask2, self.mask3))
            
            full_img = np.vstack((row1, row2))
            cv2.imshow('Full', full_img)
            
            # cv2.imshow('Mask1', mask1)
            # cv2.imshow('Mask2', mask2)
            # cv2.imshow('Mask3', mask3)
            
            
            
            if cv2.waitKey(60) == 27: 
                self.stop()
                break

    def stop(self):
        # cv2.imwrite("images/img1.png", self.img1)
        # cv2.imwrite("images/img2.png", self.img2)
        # cv2.imwrite("images/img3.png", self.img3)
        # cv2.imwrite("images/mask1.png", self.mask1)
        # cv2.imwrite("images/mask2.png", self.mask2)
        # cv2.imwrite("images/mask3.png", self.mask3)
        cv2.destroyAllWindows()

def main():
    color_calibration = ColorCalibration(color_space=cv2.COLOR_BGR2RGB)
    color_calibration.image_callback()
    color_calibration.stop()
    
if __name__ == "__main__":
    #A poco sí tilín
    main()