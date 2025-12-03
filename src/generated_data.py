import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import pandas as pd
import numpy as np
import cv2

class GenerateColors:
    def __init__(self):
        self.rgb_base_colors = {
            "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0),
            "Cyan": (0, 255, 255),
            "Green": (0, 128, 0),
            "Red": (255, 0, 0),
            "Magenta": (255, 0, 255),
            "Orange": (255, 140, 0),
            "White": (255, 255, 255),
            "Black": (0, 0, 0)
        }
        
    def generate_variations(self, rgb_color, num_variations, max_variation):
        variations = []
        for _ in range(num_variations):
            add_color = np.random.uniform(-max_variation, max_variation, 3)
            variation = np.clip(rgb_color+add_color, 0, 255).astype(int)
            variations.append(variation)
        return np.array(variations)
    
    def create_dataframe_colors(self, num_variations=9, max_variation=20):
        color_table = []
        for color, rgb in self.rgb_base_colors.items():
            variations = self.generate_variations(rgb, num_variations, max_variation)
            for var in variations:
                color_table.append([color, var])
        return pd.DataFrame(color_table, columns=['Color', 'RGB'])
    
    # def show_colors(self, df):
    #     image = np.zeros((900, 900, 3), dtype=np.uint8)
    #     for k, color in enumerate(self.rgb_base_colors.keys()):
    #         df_color = df[df['Color'] == color]
    #         for s, (index, row) in enumerate(df_color.iterrows()):
    #             x, y = k//3, k%3
    #             i, j = s//3, s%3
    #             cv2.rectangle(image, (y*300+j*100, x*300+i*100), (y*300+j*100+100, x*300+i*100+100), row["RGB"].tolist()[::-1], -1)
    #     cv2.imshow("Colors", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        
    def show_colors(self, df):
        image = np.zeros((900, 1623, 3), np.uint8)
        for k, color in enumerate(self.rgb_base_colors.keys()):
            df_color = df[df['Color'] == color]
            for s, (index, row) in enumerate(df_color.iterrows()):
                # cv2.rectangle(image, (s*100, k*100), (s*100+100, k*100+100), row["RGB"].tolist()[::-1], -1)
                cv2.rectangle(image, (s*10, k*100), (s*10+10, k*100+100), row["RGB"][::-1], -1)
        cv2.imshow("Colors", image)
        cv2.waitKey(0)
        cv2.imwrite("images/colors.png", image)
        cv2.destroyAllWindows()
        
def main():
    gc = GenerateColors()
    df = gc.create_dataframe_colors(max_variation=50, num_variations=9)
    # gc.show_colors(df)
    print(df)
    # df_colors = gc.create_dataframe_colors()
    # print(df_colors.head())

if __name__ == "__main__":
    main()