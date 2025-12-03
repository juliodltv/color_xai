import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import shap
import pandas as pd
import numpy as np
import cv2

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.colors as mcolors

import colorspacious
import colour

from itertools import combinations
import pickle

class DataAugmentation:
    def __init__(self):
        pass

    def augment_data_spaces(self, df):
        df["LCh"] = df["RGB"].apply(lambda x: colorspacious.cspace_convert(x, "sRGB255", "CIELCh")/np.array([100, 100, 360]))
        df["HSV"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2HSV)[0][0]/np.array([180, 255, 255]))
        df["HSL"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2HLS)[0][0]/np.array([180, 255, 255]))
        df["YCrCb"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2YCrCb)[0][0]/255)
        df["XYZ"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2XYZ)[0][0]/255)
        df["Lab"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2Lab)[0][0]/255)
        df["Luv"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2Luv)[0][0]/255)
        df["YUV"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2YUV)[0][0]/255)
        df["oklab"] = df["XYZ"].apply(lambda x: colour.XYZ_to_Oklab(x)+np.array([0, 0.5, 0.5]))   
        
        df["RGB"] = df["RGB"].apply(lambda x: np.uint8(x)/255)
        return df

    def separate_channels(self, df):
        df["RGB_R"] = df["RGB"].apply(lambda x: x[0])
        df["RGB_G"] = df["RGB"].apply(lambda x: x[1])
        df["RGB_B"] = df["RGB"].apply(lambda x: x[2])

        df["LCh_L"] = df["LCh"].apply(lambda x: x[0])
        df["LCh_C"] = df["LCh"].apply(lambda x: x[1])
        df["LCh_h"] = df["LCh"].apply(lambda x: x[2])

        df["HSV_H"] = df["HSV"].apply(lambda x: x[0])
        df["HSV_S"] = df["HSV"].apply(lambda x: x[1])
        df["HSV_V"] = df["HSV"].apply(lambda x: x[2])

        df["HSL_H"] = df["HSL"].apply(lambda x: x[0])
        df["HSL_S"] = df["HSL"].apply(lambda x: x[2])
        df["HSL_L"] = df["HSL"].apply(lambda x: x[1])

        df["YCrCb_Y"] = df["YCrCb"].apply(lambda x: x[0])
        df["YCrCb_Cr"] = df["YCrCb"].apply(lambda x: x[1])
        df["YCrCb_Cb"] = df["YCrCb"].apply(lambda x: x[2])
    
        df["XYZ_X"] = df["XYZ"].apply(lambda x: x[0])
        df["XYZ_Y"] = df["XYZ"].apply(lambda x: x[1])
        df["XYZ_Z"] = df["XYZ"].apply(lambda x: x[2])
    
        df["Lab_L"] = df["Lab"].apply(lambda x: x[0])
        df["Lab_a"] = df["Lab"].apply(lambda x: x[1])
        df["Lab_b"] = df["Lab"].apply(lambda x: x[2])

        df["Luv_L"] = df["Luv"].apply(lambda x: x[0])
        df["Luv_u"] = df["Luv"].apply(lambda x: x[1])
        df["Luv_v"] = df["Luv"].apply(lambda x: x[2])

        df["YUV_Y"] = df["YUV"].apply(lambda x: x[0])
        df["YUV_U"] = df["YUV"].apply(lambda x: x[1])
        df["YUV_V"] = df["YUV"].apply(lambda x: x[2])

        df["oklab_L"] = df["oklab"].apply(lambda x: x[0])
        df["oklab_a"] = df["oklab"].apply(lambda x: x[1])
        df["oklab_b"] = df["oklab"].apply(lambda x: x[2])
        
        df = df.drop(columns=["RGB", "LCh","HSV", "HSL", "YCrCb", "XYZ", "Lab", "Luv", "YUV", "oklab"])
        return df
    
def main():
    
    df = pd.read_pickle("dataframes/dataset.pkl")
    da = DataAugmentation()
    df = da.augment_data_spaces(df)
    spaces = df.columns[1:].to_list()
    df = da.separate_channels(df)
    chanels = df.columns[1:].to_list()

    color_to_int = {
        "blue": 0,
        "yellow": 1,
        "cyan": 2,
        "green": 3,
        "red": 4,
        "magenta": 5,
        "orange": 6,
        "white": 7,
        "black": 8,
    }
    int_to_color = {v: k for k, v in color_to_int.items()}
    
    colors = list(int_to_color.values())
    colors[-2] = "gray"
    cmap_order = mcolors.ListedColormap(colors)
    
    spaces_channels = {space: [chanels[i+j*3] for i in range(3)] for j, space in enumerate(spaces)}
    combination_spaces = {space: {} for space in spaces}
    for space, channels in spaces_channels.items():
        for i in range(1,3+1):
            combination = np.array(list(combinations(channels, i)))
            num_combinations = np.array(list(combinations(range(1, 3+1), i)))            
            for comb, num in zip(combination, num_combinations):
                X = df[comb]
                y = df["color"].map(color_to_int)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                model = SVC(kernel="rbf", decision_function_shape="ovr", C=100.0, gamma=1.0, probability=True)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                print(f"Channels: {comb}{num}, Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                x_sumary = shap.sample(X, 10, random_state=42)
                explainer = shap.KernelExplainer(model.predict_proba, data=x_sumary, link="logit")
                shap_values = explainer.shap_values(X_test)
                shap_values = [np.where(values>0, values, 0) for values in shap_values]
                combination_spaces[space][str(num)] = shap_values
        print()
    
    # print(combination_spaces.keys())
    # print(combination_spaces["HSL"].keys())
    
    with open("shap_values/shap_values_n.pkl", "wb") as f:
        pickle.dump(combination_spaces, f)
        
    # shap.summary_plot(shap_values, X_test, class_names=['']*10, max_display=10*3, color=cmap_order)

if __name__ == "__main__":
    main()