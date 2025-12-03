import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

#picke
df = pd.read_pickle("dataframes/shap_values.pkl")
# df = pd.read_csv("dataframes/chatgpt.csv")

spaces = {
    "RGB": [
        "RGB_R", 
        "RGB_G", 
        "RGB_B"
    ],
    "LCh": [
        "LCh_L", 
        "LCh_C", 
        "LCh_h"
    ],
    "HSV": [
        "HSV_H", 
        "HSV_S", 
        "HSV_V"
    ],
    "HSL": [
        "HSL_H", 
        "HSL_S", 
        "HSL_L"
    ],
    "YCrCb": [
        "YCrCb_Y", 
        "YCrCb_Cr", 
        "YCrCb_Cb"
    ],
    "XYZ": [
        "XYZ_X", 
        "XYZ_Y", 
        "XYZ_Z"
    ],
    "Lab": [
        "Lab_L", 
        "Lab_a", 
        "Lab_b"
    ],
    "Luv": [
        "Luv_L", 
        "Luv_u", 
        "Luv_v"
    ],
    "YUV": [
        "YUV_Y", 
        "YUV_U", 
        "YUV_V"
    ],
    "oklab": [
        "oklab_L", 
        "oklab_a", 
        "oklab_b"
    ]
}


colors = {
    "blue": 0,
    "yellow": 1,
    "cyan": 2,
    "green": 3,
    "red": 4,
    "magenta": 5,
    "orange": 6,
    "white": 7,
    "black": 8
}

print(df.head(10))
sums = 0
val_spaces = [0 for _ in range(len(spaces))]

spaces_value = {
    "RGB": 0,
    "LCh": 0,
    "HSV": 0,
    "HSL": 0,
    "YCrCb": 0,
    "XYZ": 0,
    "Lab": 0,
    "Luv": 0,
    "YUV": 0,
    "oklab": 0
}


num_channels = 3
list_channels = []
dict_channels = defaultdict(lambda: 0)
for space in spaces:
    for color, c in colors.items():
        # if color == "black" or color == "white": continue
        combi = combinations(range(3), num_channels)
        for indices in combi:
            if len(indices) == 1:
                single_value = df[spaces[space][indices[0]]].iloc[c]
                list_channels.append((spaces[space][indices[0]], color, single_value))
                # dict_channels[spaces[space][indices[0]]] += single_value
                dict_channels[spaces[space][indices[0]]] = max(dict_channels[spaces[space][indices[0]]], single_value)
            elif len(indices) == 2:
                max_value = max(df[spaces[space][indices[0]]].iloc[c], df[spaces[space][indices[1]]].iloc[c])
                list_channels.append((spaces[space][indices[0]], spaces[space][indices[1]], color, max_value))
                dict_channels[(spaces[space][indices[0]], spaces[space][indices[1]])] += max_value
            else:
                max_value = max(df[spaces[space][indices[0]]].iloc[c], df[spaces[space][indices[1]]].iloc[c], df[spaces[space][indices[2]]].iloc[c])
                list_channels.append((spaces[space][indices[0]], spaces[space][indices[1]], spaces[space][indices[2]], color, max_value))
                dict_channels[(spaces[space][indices[0]], spaces[space][indices[1]], spaces[space][indices[2]])] += max_value

list_channels = sorted(list_channels, key=lambda x: x[num_channels+1], reverse=True)

for item in list_channels:
    print(item)

sorted_items_desc = sorted(dict_channels.items(), key=lambda item: item[1], reverse=True)
for k, v in sorted_items_desc:
    print(k, f"{v:.4f}")
    
# for k, v in dict_channels.items():
#     print(k, v)
    
# for item in list_channels:
#     if item[0] == "HSL_H" and item[1] == "HSL_S":
#         print(item)

# For all colors

# usar el dbscan para una linea, closets point
# SHAP F selection

# spaces_value = dict(sorted(spaces_value.items(), key=lambda item: item[1], reverse=True))
# for k, v in spaces_value.items():
#     print(k, f"{v:.4f}")
    
    