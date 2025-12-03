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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import colorspacious
import colour

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
        # df["HSV_V"] = df["HSV"].apply(lambda x: x[2])

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
    # df = df[(df["color"] != "white") & (df["color"] != "black")]
    da = DataAugmentation()
    df = da.augment_data_spaces(df)
    # print(df.head())
    # df.to_pickle("dataframes/color_spaces.pkl")
    # quit()
    spaces = df.columns[1:].to_list()
    df = da.separate_channels(df)
    chanels = df.columns[1:].to_list()
    print(chanels)
    # quit()
    print(df.head())
    # quit()

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

    y = df["color"].map(color_to_int)
    X = df.drop(columns=["color"])
    
    # # To search for the best hyperparameters
    # param_grid = {
    #     "C": [0.1, 1, 10, 100, 100],
    #     "gamma": [1, 0.1, 0.01, 0.001],
    #     "kernel": ["rbf", "poly", "sigmoid"]
    # }
    # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
    # grid.fit(X, y)
    # print(grid.best_params_)
    # quit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = SVC(kernel="rbf", decision_function_shape="ovr", C=100.0, gamma=1.0, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Report")
    print(classification_report(y_test, y_pred, target_names=int_to_color.values()))
    
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    
    print("Accuracy")
    print(f"{accuracy:.4f}")
    quit()
    colors = [
        "blue",
        "yellow",
        "cyan",
        "green",
        "red",
        "magenta",
        "orange",
        "gray",
        "black",
    ]
    
    white_and_black = True
    if not white_and_black:
        colors = colors[:-2]
    cmap_order = mcolors.ListedColormap(colors)
    colors = list(int_to_color.values())
    
    # # Crear malla para el fondo
    # x_min, x_max = 0, 1
    # y_min, y_max = 0, 1
    # step = 0.02
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
    #                     np.arange(y_min, y_max, step))

    # Predecir en cada punto de la malla
    # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # # Graficar el fondo con los límites de decisión
    # plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_colors)

    # # Graficar también los puntos de datos
    # scatter = plt.scatter(X.values[:, 0], X.values[:, 1], c=y, edgecolors='k', cmap=cmap_colors)
    # legend1 = plt.legend(handles=scatter.legend_elements()[0], labels=colors, title="Clases")
    # plt.xlabel('canal 1')
    # plt.ylabel('canal 2')
    # plt.title('SVD')
    # plt.show()
    # quit()
    
    # x_sumary = shap.kmeans(X, 9)
    x_sumary = shap.sample(X, 10, random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, data=x_sumary, link="logit")
    # shap_values = explainer.shap_values(X_test, nsamples=100, l1_reg="num_features(10)", random_state=42)
    shap_values = explainer.shap_values(X_test)
    
    shap_values = [np.where(values>0, values, 0) for values in shap_values]
    normalized = [np.zeros_like(shap_values[0]) for i in range(len(shap_values))]
    
    shap.summary_plot(shap_values, X_test, class_names=['']*10, max_display=10*3, color=cmap_order)
    # quit()
    # print(len(shap_values))
    for color in range(len(shap_values)):
        for space in range(len(spaces)):
            suma = shap_values[color][:,space*3:space*3+3].sum()
            normalized[color][:,space*3:space*3+3] = shap_values[color][:,space*3:space*3+3]/suma
    
    data = {
        "color": [],
        "RGB_R": [],
        "RGB_G": [],
        "RGB_B": [],
        "LCh_L": [],
        "LCh_C": [],
        "LCh_h": [],
        "HSV_H": [],
        "HSV_S": [],
        "HSV_V": [],
        "HSL_H": [],
        "HSL_S": [],
        "HSL_L": [],
        "YCrCb_Y": [],
        "YCrCb_Cr": [],
        "YCrCb_Cb": [],
        "XYZ_X": [],
        "XYZ_Y": [],
        "XYZ_Z": [],
        "Lab_L": [],
        "Lab_a": [],
        "Lab_b": [],
        "Luv_L": [],
        "Luv_u": [],
        "Luv_v": [],
        "YUV_Y": [],
        "YUV_U": [],
        "YUV_V": [],
        "oklab_L": [],
        "oklab_a": [],
        "oklab_b": [],
    }
    
    for color in range(len(normalized)):
        data["color"].append(colors[color])
        max_value = normalized[color].sum(axis=0)
        for i, key in enumerate(data.keys()):
            if key != "color":
                data[key].append(max_value[i-1])
    
    print(data)
    
    results = pd.DataFrame(data)
    # results = results.T
    print(results.to_latex(index=False, float_format="%.4f"))
    
    quit()
    
    if not white_and_black:
        normalized = normalized[:-2]
    shap.summary_plot(normalized, X_test, class_names=['']*10, max_display=10*3, color=cmap_order)
    print(normalized)

if __name__ == "__main__":
    main()