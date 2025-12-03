import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

color_space_name = "oklab"

class GraphColorSpace:
    def __init__(self, color_space, color_space_name):
        self.color_space = color_space
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel(color_space_name[2])
        self.ax.set_ylabel(color_space_name[3])
        self.ax.set_zlabel(color_space_name[4])
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_zlim([0, 1])
        self.ax.set_title(color_space_name)
        # self.ax.view_init(30, 30)
        self.ax.view_init(45, 135)
        
        
    def plot_color_space(self, cmap, num):
        # self.ax.scatter(self.color_space[:, 0], self.color_space[:, 1], self.color_space[:, 2], c=self.color_space, marker='o', alpha=1)
        self.ax.scatter(self.color_space[:, 0], self.color_space[:, 1], self.color_space[:, 2], c=range(num), cmap=cmap, marker='o', alpha=1.0, edgecolors='k')
        plt.show()

def main():
    df = pd.read_pickle("dataframes/color_spaces.pkl")
    df.sort_values(by="color", inplace=True)
    num = df["color"].size
    color_list = df["color"].unique()
    cmap = mcolors.ListedColormap(color_list)
    color_space = df[color_space_name].values
    color_space = np.array([np.array(x) for x in color_space])
    gcs = GraphColorSpace(color_space, color_space_name)
    gcs.plot_color_space(cmap, num)
    

if __name__ == "__main__":
    main()