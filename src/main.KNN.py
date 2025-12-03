import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import shap
import pandas as pd
import numpy as np
import cv2

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import colorspacious
# import colour

class DataAugmentation:
    def __init__(self):
        pass

    def augment_data_spaces(self, df):
        df["LCh"] = df["RGB"].apply(lambda x: colorspacious.cspace_convert(x, "sRGB255", "CIELCh")/np.array([100, 100, 360]))
        df["HSV"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2HSV)[0][0]/np.array([180, 255, 255]))
        df["HSL"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2HLS)[0][0]/np.array([180, 255, 255]))
        df["YCrCb"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2YCrCb)[0][0]/255)
        df["XYZ"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2XYZ)[0][0]/255)
        # df["oklab"] = df["XYZ"].apply(lambda x: colour.XYZ_to_Oklab(x))    
        df["Lab"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2Lab)[0][0]/255)
        df["Luv"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2Luv)[0][0]/255)
        df["YUV"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2YUV)[0][0]/255)
        
        df["RGB"] = df["RGB"].apply(lambda x: np.uint8(x)/255)
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
        # df["oklab_L"] = df["oklab"].apply(lambda x: x[0])
        # df["oklab_a"] = df["oklab"].apply(lambda x: x[1])
        # df["oklab_b"] = df["oklab"].apply(lambda x: x[2])

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

        
        df = df.drop(columns=["RGB", "LCh","HSV", "HSL", "YCrCb", "XYZ", "Lab", "Luv", "YUV"])
        return df
    
def main():
    
    df = pd.read_pickle("real_data_regras/video.pkl")
    # df = df[(df["Color"] == "White") | (df["Color"] == "Black")]
    # df = df[(df["Color"] != "White") & (df["Color"] != "Black")]
    da = DataAugmentation()
    df = da.augment_data_spaces(df)
    # df.to_pickle("dataframes/color_spaces.pkl")
    df = da.separate_channels(df)
    # print(df.head())
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
        "Black": 8,
    }

    int_to_color = {v: k for k, v in color_to_int.items()}

    y = df["Color"].map(color_to_int)
    X = df.drop(columns=["Color"])
    
    # # To search for the best hyperparameters
    # param_grid = {
    #     "C": [0.1, 1, 10, 100, 100],
    #     "gamma": [1, 0.1, 0.01, 0.001],
    #     "kernel": ["rbf", "poly", "sigmoid"]
    # }
    # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=, cv=5)
    # grid.fit(X, y)
    # print(grid.best_params_)
    # quit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=35)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Report")
    print(classification_report(y_test, y_pred, target_names=int_to_color.values()))
    
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    
    print("Accuracy")
    print(accuracy)
    
    quit()
    # colors = [x.lower() for x in list(color_to_int.keys())]
    # cmap_colors = mcolors.ListedColormap(colors)
    # colors_order = [
    #     "red",
    #     "blue",
    #     "orange",
    #     "green",
    #     "yellow",
    #     "cyan",
    #     "magenta",
    #     "black",
    #     "gray",
    # ]
    # cmap_order = mcolors.ListedColormap(colors_order)
    
    # x_sumary = shap.sample(X, 10)
    # explainer = shap.Explainer(knn, data=x_sumary, link="logit")
    # shap_values = explainer.shap_values(X, nsamples=100)
    # # shap_values = [np.where(values>0, values, 0) for values in shap_values]
    # shap.summary_plot(shap_values, X, class_names=colors, max_display=9*3, color=cmap_order)
    # quit()
    
    # shap.plots.heatmap(shap_values)

    
    def f(x):
        return knn.predict(x)
    
    med = X_train.median().values.reshape(1, X_train.shape[1])
    
    # explainer = shap.Explainer(f, med)
    # shap_values = explainer.shap_values(X_test)
    
    x_sumary = shap.sample(X, 100)
    explainer = shap.KernelExplainer(knn.predict_proba, data=x_sumary, link="logit")
    shap_values = explainer.shap_values(X_test, nsamples=100, l1_reg="num_features(10)")
    
    # shap.plots.beeswarm(shap_values)
    shap.summary_plot(shap_values, X, class_names=colors, max_display=9*3, color=cmap_order)
    # y_pred = knn.predict_proba(X_test.values)
    # print(y_pred)
    
    quit()
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    model = SVC(kernel="rbf", decision_function_shape="ovr", C=100.0, gamma=1.0, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Report")
    print(classification_report(y_test, y_pred, target_names=int_to_color.values()))
    
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    
    print("Accuracy")
    print(accuracy)
    # quit()
    
    colors = [x.lower() for x in list(color_to_int.keys())]
    cmap_colors = mcolors.ListedColormap(colors)
    colors_order = [
        "red",
        "blue",
        "orange",
        "green",
        "yellow",
        "cyan",
        "magenta",
        "black",
        "gray",
    ]
    cmap_order = mcolors.ListedColormap(colors_order)
    
    # Crear malla para el fondo
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                        np.arange(y_min, y_max, step))

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
    
    # x_sumary = shap.kmeans(X, 100)
    x_sumary = shap.sample(X, 100)
    explainer = shap.KernelExplainer(model.predict_proba, data=x_sumary, link="logit")
    # explainer = shap.KernelExplainer(model.predict, data=x_sumary)
    #link="logit"
    shap_values = explainer.shap_values(X_test, nsamples=100, l1_reg="num_features(10)")
    # shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X, class_names=colors, max_display=9*3, color=cmap_order)
    
    grupos = {
        'RGB': ['RGB_R', 'RGB_G', 'RGB_B'],
        "LCh": ['LCh_L', 'LCh_C', 'LCh_h'],
        'HSV': ['HSV_H', 'HSV_S', 'HSV_V'],
        "HSL": ['HSL_H', 'HSL_S', 'HSL_L'],
        "YCrCb": ['YCrCb_Y', 'YCrCb_Cr', 'YCrCb_Cb'],
        "XYZ": ['XYZ_X', 'XYZ_Y', 'XYZ_Z'],
        "Lab": ['Lab_L', 'Lab_a', 'Lab_b'],
        "Luv": ['Luv_L', 'Luv_u', 'Luv_v'],
        "YUV": ['YUV_Y', 'YUV_U', 'YUV_V'],
        # Añade aquí otros grupos según sea necesario
    }
        
    # Para simplificar y como ejemplo, calcularé la importancia de un grupo como el promedio de las importancias de sus características
    importancias_grupos = {}

    # Recorrer cada grupo de características y calcular su importancia promedio
    for nombre_grupo, caracteristicas_grupo in grupos.items():
        # Asegúrate de ajustar el índice de shap_values según la clase de interés, si es necesario
        indices = [X.columns.get_loc(c) for c in caracteristicas_grupo]
        importancias_grupo = np.mean([np.abs(shap_values[clase][:, indices]).mean(0) for clase in range(len(shap_values))], axis=0)
        importancias_grupos[nombre_grupo] = np.mean(importancias_grupo)

    # Mostrar las importancias de los grupos
    for grupo, importancia in importancias_grupos.items():
        print(f"Grupo {grupo}: Importancia media {importancia}")

if __name__ == "__main__":
    main()