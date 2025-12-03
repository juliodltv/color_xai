import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import pandas as pd
import numpy as np
import cv2

class VideoDF:
    def __init__(self):
        self.test_points = [110, 200]   # roi width, roi height for each color
        self.init_points = {            # initial points for each color
            "Blue": [400, 325],
            "Yellow": [8, 564],
            "Cyan": [195, 578],
            "Green": [194, 316],
            "Red": [9, 305],
            "Magenta": [587, 326],
            "Orange": [400, 595],
            "White": [321, 65],
            "Black": [575, 575]
        }
        self.points = self.roi_points()
    
    def roi_points(self):
        points = {}
        for key in self.init_points.keys():
            points[key] = [
                self.init_points[key],
                [self.init_points[key][0]+self.test_points[0]//2, self.init_points[key][1]],
                [self.init_points[key][0]+self.test_points[0], self.init_points[key][1]],
                [self.init_points[key][0], self.init_points[key][1]+self.test_points[1]//2],
                [self.init_points[key][0]+self.test_points[0]//2, self.init_points[key][1]+self.test_points[1]//2],
                [self.init_points[key][0]+self.test_points[0], self.init_points[key][1]+self.test_points[1]//2],
                [self.init_points[key][0], self.init_points[key][1]+self.test_points[1]],
                [self.init_points[key][0]+self.test_points[0]//2, self.init_points[key][1]+self.test_points[1]],
                [self.init_points[key][0]+self.test_points[0], self.init_points[key][1]+self.test_points[1]]
            ]
        return points
    
    def create_dataframe_video(self, path):
        df = pd.DataFrame(columns=["Color", "RGB"])
        cam = cv2.VideoCapture(path)
        count = 0
        while True:
            ret, frame = cam.read()
            if not ret: break
            frame = frame[0:800, :, :]
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 27: break
            count += 1
            if count % 30 == 0:
                for key in self.points.keys():
                    for point in self.points[key]:
                        df.loc[len(df)] = [key, frame[point[1], point[0]].tolist()[::-1]]
            
        cam.release()
        cv2.destroyAllWindows()
        return df
    
    def create_dataframe_images(self, path):
        df = pd.DataFrame(columns=["Color", "RGB"])
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            for key in self.points.keys():
                for point in self.points[key]:
                    df.loc[len(df)] = [key, img[point[1], point[0]].tolist()[::-1]]
        return df

def main():
    path = "real_data/video.mp4"
    df = VideoDF().create_dataframe_video(path)
    df.to_pickle("real_data/video2.pkl")
    df = pd.read_pickle("real_data/video2.pkl")
    print(df)

if __name__ == "__main__":
    main()