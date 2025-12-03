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
        self.points = {
            "blue": [[72, 55], [297, 61], [103, 218], [259, 222], [102, 393], [253, 392]],
            "yellow": [[69, 137], [293, 145], [103, 303], [254, 309], [98, 480], [254, 479]],
            "cyan": [[164, 92], [310, 177], [162, 260], [237, 344], [84, 427], [270, 517]],
            "green": [[89, 92], [272, 179], [87, 258], [272, 346], [194, 429], [195, 518]],
            "red": [[50, 89], [236, 95], [46, 170], [234, 175], [198, 257], [194, 344]],
            "magenta": [[122, 256], [121, 341], [120, 429], [237, 430], [116, 516], [236, 515]],
            "orange": [[323, 235], [327, 275], [329, 344], [328, 412], [328, 462], [327, 528]],
            "white": [[27, 230], [27, 318], [27, 365], [22, 438], [25, 495], [25, 554]],
            "black": [[107, 116], [256, 121], [142, 281], [217, 284], [140, 454], [217, 455]],
        }
        
        self.selected_points = []
    
    def create_dataframe_video(self, path):
        df = pd.DataFrame(columns=["color", "RGB"])
        cam = cv2.VideoCapture(path)
        count = 0
        while True:
            ret, frame = cam.read()
            if not ret: break
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 27: break
            count += 1
            if count % 5 == 0 and count>15:
                for key in self.points.keys():
                    for point in self.points[key]:
                        df.loc[len(df)] = [key, frame[point[1], point[0]].tolist()[::-1]]
            if count % 10 == 0 and count>15:
                print(frame.shape)
                pts1 = np.float32([[32, 33], [18, 538], [318, 536], [328, 45]])
                pts2 = np.float32([[0,0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]], [frame.shape[1], 0]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                frame = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
                # frame = frame[33:540, 23:330]
                cv2.imwrite(f"real_data_regras/frames/frame{count}.jpg", frame)
            
        cam.release()
        cv2.destroyAllWindows()
        return df
    
    def get_points(self, path):
        # cv2.namedWindow("Frame")
        cam = cv2.VideoCapture(path)
        while True:
            ret, frame = cam.read()
            if not ret: break
            
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            
            cv2.imshow("Frame", frame)
            cv2.setMouseCallback("Frame", self.click_event)
            cv2.waitKey(0)
            break
        cam.release()
        cv2.destroyAllWindows()
            
    
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append([x, y])
            print(self.selected_points)
        

def main():
    videodf = VideoDF()
    path = "real_data_regras/thechosenone.mp4"
    # videodf.get_points(path)
    df = videodf.create_dataframe_video(path)
    # print(df.head())
    # print(df.tail())
    # path = "real_data/video.mp4"
    # df = VideoDF().create_dataframe_video(path)
    df.to_pickle("real_data_regras/video2.pkl")
    df = pd.read_pickle("real_data_regras/video2.pkl")
    print(df)

if __name__ == "__main__":
    main()