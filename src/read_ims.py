import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import pandas as pd
import numpy as np
import cv2

class ImageDF:
    def __init__(self):
        self.points = {
            "Blue": [[215, 195], [322, 302], [215, 409]],
            "Yellow": [[479, 298], [586, 193], [588, 410]],
            "Cyan": [[576, 401,], [467, 296], [334, 295], [575, 187]],
            "Green": [[224, 198], [333, 307], [225, 415]],
            "Red": [[225, 402], [578, 414]],
            "Magenta": [[225, 187], [575, 198]],
            "Orange": [[400, 300]] 
        }
    
    def create_dataframe_images(self, path):
        df = pd.DataFrame(columns=["Color", "RGB"])
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            for key in self.points.keys():
                for point in self.points[key]:
                    df.loc[len(df)] = [key, img[point[1], point[0]].tolist()[::-1]]
        return df

def main():
    path = "iluminations"
    df = ImageDF().create_dataframe_images(path)
    print(df)

if __name__ == "__main__":
    main()