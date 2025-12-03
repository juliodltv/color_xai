import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import lightgbm as lgb
import shap
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import colorspacious
import colour


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

    def show_colors(self, df):
        image = np.zeros((700, 1623, 3), np.uint8)
        for k, color in enumerate(self.rgb_base_colors.keys()):
            df_color = df[df['Color'] == color]
            for s, (index, row) in enumerate(df_color.iterrows()):
                cv2.rectangle(image, (s*10, k*100), (s*10+10, k*100+100), row["RGB"][::-1], -1)
        cv2.imshow("dataset", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

class DataAugmentation:
    def __init__(self):
        pass

    def augment_data_spaces(self, df):
        df["LCh"] = df["RGB"].apply(lambda x: np.int32(np.round(colorspacious.cspace_convert(x, "sRGB255", "CIELCh"))))     # [0-100, 0-100, 0-360]
        df["HSV"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2HSV_FULL)[0][0])                  # [0-255, 0-255, 0-255]
        df["HSL"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2HLS_FULL)[0][0])                  # [0-255, 0-255, 0-255]
        df["YCrCb"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2YCrCb)[0][0])                   # [0-255, 0-255, 0-255]
        df["XYZ"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2XYZ)[0][0])                       # [0-255, 0-255, 0-255]
        df["oklab"] = df["XYZ"].apply(lambda x: colour.XYZ_to_Oklab(x/255.0))     # [0-255, 0-255, 0-255]
        df["Lab"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2Lab)[0][0])                       # [0-255, 0-255, 0-255]
        df["Luv"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2Luv)[0][0])                       # [0-255, 0-255, 0-255]
        df["YUV"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2YUV)[0][0])                       # [0-255, 0-255, 0-255]
        return df

    def separate_channels(self, df):
        df["RGB_R"] = df["RGB"].apply(lambda x: x[0])
        df["RGB_G"] = df["RGB"].apply(lambda x: x[1])
        df["RGB_B"] = df["RGB"].apply(lambda x: x[2])

        # CIELCh
        df["LCh_L"] = df["LCh"].apply(lambda x: x[0])
        df["LCh_C"] = df["LCh"].apply(lambda x: x[1])
        df["LCh_h"] = df["LCh"].apply(lambda x: x[2])
        
        # Oklab
        df["oklab_L"] = df["oklab"].apply(lambda x: x[0])
        df["oklab_a"] = df["oklab"].apply(lambda x: x[1])
        df["oklab_b"] = df["oklab"].apply(lambda x: x[2])

        df["HSV_H"] = df["HSV"].apply(lambda x: x[0])
        df["HSV_S"] = df["HSV"].apply(lambda x: x[1])
        df["HSV_V"] = df["HSV"].apply(lambda x: x[2])
        df["HSL_H"] = df["HSL"].apply(lambda x: x[0])
        df["RGB_G"] = df["RGB"].apply(lambda x: x[1])
        df["RGB_B"] = df["RGB"].apply(lambda x: x[2])
        df["HSL_S"] = df["HSL"].apply(lambda x: x[2])
        df["HSL_L"] = df["HSL"].apply(lambda x: x[1])
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

        df = df.drop(columns=["RGB", "LCh","HSV", "HSL", "YCrCb", "XYZ", "Lab", "Luv", "YUV", "oklab"])
        return df
    
def main():
    
    gc = GenerateColors()
    df = pd.read_pickle("real_data_regras/video.pkl")
    # df = df[(df["Color"] == "White") | (df["Color"] == "Black")]
    # df = df[(df["Color"] != "White") & (df["Color"] != "Black")]
    # gc.show_colors(df)
    da = DataAugmentation()
    df = da.augment_data_spaces(df)
    # min_value = df['oklab'].apply(lambda x: x[2]).min()
    # max_value = df['oklab'].apply(lambda x: x[2]).max()
    # print(min_value, max_value)
    # print(df.head())
    # quit()
    df = da.separate_channels(df)

    # Normalize data
    color = df["Color"]
    LCh_L = df["LCh_L"]/100.0
    LCh_C = df["LCh_C"]/100.0
    LCh_h = df["LCh_h"]/360.0
    
    oklab_L = df["oklab_L"]
    oklab_a = df["oklab_a"]
    oklab_b = df["oklab_b"]
    
    
    df = df.drop(columns=["Color", "LCh_L", "LCh_C", "LCh_h", "oklab_L", "oklab_a", "oklab_b"])
    # df = df.drop(columns=["Color", "LCh_L", "LCh_C", "LCh_h"])
    df = df/255.0
    df.insert(0, "oklab_a", oklab_a)
    df.insert(0, "oklab_b", oklab_b)
    df.insert(0, "oklab_L", oklab_L)
    df.insert(0, "LCh_h", LCh_h)
    df.insert(0, "LCh_C", LCh_C)
    df.insert(0, "LCh_L", LCh_L)
    df.insert(0, "Color", color)
    
    spaces = [
        ["HSL_H", "HSL_S"],
        ["HSL_H", "HSL_L"],
        ["HSL_S", "HSL_L"],
        ["HSV_H", "HSV_S"],
        ["HSV_H", "HSV_V"],
        ["HSV_S", "HSV_V"],
        ["Lab_L", "Lab_a"],
        ["Lab_L", "Lab_b"],
        ["Lab_a", "Lab_b"],
        ["LCh_C", "LCh_L"],
        ["LCh_h", "LCh_L"],
        ["LCh_h", "LCh_C"],
        ["Luv_L", "Luv_u"],
        ["Luv_L", "Luv_v"],
        ["Luv_u", "Luv_v"],
        ["oklab_L", "oklab_a"],
        ["oklab_L", "oklab_b"],
        ["oklab_a", "oklab_b"],
        ["RGB_R", "RGB_G"],
        ["RGB_R", "RGB_B"],
        ["RGB_G", "RGB_B"],
        ["XYZ_X", "XYZ_Y"],
        ["XYZ_X", "XYZ_Z"],
        ["XYZ_Y", "XYZ_Z"],
        ["YCrCb_Y", "YCrCb_Cr"],
        ["YCrCb_Y", "YCrCb_Cb"],
        ["YCrCb_Cr", "YCrCb_Cb"],
        ["YUV_Y", "YUV_U"],
        ["YUV_Y", "YUV_V"],
        ["YUV_U", "YUV_V"],
    ]
    
    titles = [
        "HSL_HS",
        "HSL_HL",
        "HSL_SL",
        "HSV_HS",
        "HSV_HV",
        "HSV_SV",
        "Lab_La",
        "Lab_Lb",
        "Lab_ab",
        "LCh_CL",
        "LCh_hL",
        "LCh_hC",
        "Luv_Lu",
        "Luv_Lv",
        "Luv_uv",
        "oklab_La",
        "oklab_Lb",
        "oklab_ab",
        "RGB_RG",
        "RGB_RB",
        "RGB_GB",
        "XYZ_XY",
        "XYZ_XZ",
        "XYZ_YZ",
        "YCrCb_YCr",
        "YCrCb_YCb",
        "YCrCb_CrCb",
        "YUV_YU",
        "YUV_YV",
        "YUV_UV",
    ]
    
    labels = [
        ["H", "S"],
        ["H", "L"],
        ["S", "L"],
        ["H", "S"],
        ["H", "V"],
        ["S", "V"],
        ["L", "a"],
        ["L", "b"],
        ["a", "b"],
        ["C", "L"],
        ["h", "L"],
        ["h", "C"],
        ["Lu", "L"],
        ["Lv", "L"],
        ["u", "v"],
        ["L", "a"],
        ["L", "b"],
        ["a", "b"],
        ["R", "G"],
        ["R", "B"],
        ["G", "B"],
        ["X", "Y"],
        ["X", "Z"],
        ["Y", "Z"],
        ["Y", "Cr"],
        ["Y", "Cb"],
        ["Cr", "Cb"],
        ["Y", "U"],
        ["Y", "V"],
        ["U", "V"],
    ]
    
    # fig, ax = plt.subplots(figsize=(12, 8))
    # for color in df["Color"].unique():
    #     df_color = df[df["Color"] == color]
    #     plt.scatter(df_color["LCh_h"].values, df_color["XYZ_Z"].values, label=color, color=color.lower() if color != "White" else "gray")
    # # plt.xlabel(labels[i][0])
    # # plt.ylabel(labels[i][1])
    # # plt.title(titles[i]+" color space")
    # plt.show()
    # quit()
    
    # for i in range(len(spaces)):
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     for color in df["Color"].unique():
    #         df_color = df[df["Color"] == color]
    #         plt.scatter(df_color[spaces[i][0]].values, df_color[spaces[i][1]].values, label=color, color=color.lower() if color != "White" else "gray")
    #     plt.xlabel(labels[i][0])
    #     plt.ylabel(labels[i][1])
    #     plt.title(titles[i]+" color space")
    #     # plt.legend()
    #     # plt.savefig(f"color_spaces/2D/{titles[i]}.png")
    #     plt.show()
    # quit()
    # data = {
    #     "Color": [],
    #     "Hue min": [],
    #     "Hue max": [],
    #     "Light min": [],
    #     "Light max": []  
    # }
    
    # for color in df["Color"].unique():
    #     data["Color"].append(color)
    #     df_color = df[df["Color"] == color]
    #     data["Hue min"].append(df_color["HSL_H"].min())
    #     data["Hue max"].append(df_color["HSL_H"].max())
    #     data["Light min"].append(df_color["HSL_L"].min())
    #     data["Light max"].append(df_color["HSL_L"].max())
    
    # df_data = pd.DataFrame(data)
    # print(df_data.to_latex(index=False, float_format="%.4f"))
    
    # np.random.seed(42)
    # df_latex = df.sample(n=5)
    # df_latex = df_latex.iloc[:, :5]
    # print(df_latex.to_latex(index=False, float_format="%.2f"))
    # quit()

    color_to_int = {
        "Blue": 0,
        "Yellow": 1,
        "Cyan": 2,
        "Green": 3,
        "Red": 4,
        "Magenta": 5,
        "Orange": 6,
        "White": 7,
        "Black": 7,
    }

    int_to_color = {v: k for k, v in color_to_int.items()}

    y = df["Color"].map(color_to_int)
    X = df.drop(columns=["Color"])
    
    # for color in color_to_int.keys():
    #     df_filter = df[df["Color"] == color]
        
    #     min_h = df_filter["HSL_H"].min()
    #     max_h = df_filter["HSL_H"].max()
        
    #     min_l = df_filter["HSL_L"].min()
    #     max_l = df_filter["HSL_L"].max()
        
    #     print(f"Color: {color}, Min H: {min_h:.4f}, Max H: {max_h:.4f}", end=" ")
    #     print(f"Min L: {min_l:.4f}, Max L: {max_l:.4f}")
    
    # quit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    d_train = lgb.Dataset(X_train, label=y_train)
    d_test = lgb.Dataset(X_test, label=y_test)

    params = {
        "num_threads": 8,
        "max_bin": 512,
        "learning_rate": 0.05,
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_leaves": 10,
        "verbose": -1,
        "min_data": 100,
        "num_class": 8,
        "early_stopping_round": 100,
        "max_depth": 2,
    }

    model = lgb.train(
        params,
        d_train,
        10000,
        valid_sets=[d_test],
    )
    
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("Report")
    print(classification_report(y_test, y_pred, target_names=int_to_color.values()))
    
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    
    print("Accuracy")
    print(accuracy)

    # # print(model.num_trees())
    
    # ax = lgb.plot_tree(model, tree_index=1000, figsize=(20, 8), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
    # plt.show()
    # # quit()
    
    
    # # Generar datos para hue y light
    # hue = np.linspace(0, 1, 1_000)  # Ajusta según la cantidad de puntos deseados
    # light = np.linspace(0, 1, 1_000)  # Ajusta según la cantidad de puntos deseados

    # # Preparar el conjunto de características combinando hue y light
    # features = np.array([[h, l] for h in hue for l in light])

    # # Realizar predicciones con el modelo
    # y_pred = model.predict(features, num_iteration=model.best_iteration)
    # y_pred = np.argmax(y_pred, axis=1)
    # y_pred = [int_to_color[i] for i in y_pred]

    # current_color = None
    
    # for hue_value, color in zip(hue, y_pred):
    #     if current_color is None:
    #         current_color = color
    #         print(f"Hue: {hue_value[0]:.4f}, Predicted Color: {color}")
    #     elif current_color != color:
    #         # df_filter = df[df["Color"] == current_color]
    #         # print(df_filter.head())
    #         current_color = color
    #         print(f"Hue: {hue_value[0]:.4f}, Predicted Color: {color}")
    
    quit()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    

    colors = ["blue", "yellow", "cyan", "green", "red", "magenta", "orange", "white", "black"]
    # colors = ["white", "black"]
    # colors_order = ["cyan", "red", "blue", "orange", "green", "yellow", "magenta", "gray", "black"]
    colors_order = [
        "green",
        "orange",
        "yellow",
        "cyan",
        "blue",
        "red",
        "magenta",
        "black",
        "gray",
    ]
    cmap = mcolors.ListedColormap(colors_order)

    shap.summary_plot(shap_values, X, class_names=colors, max_display=100, color=cmap)

if __name__ == "__main__":
    main()