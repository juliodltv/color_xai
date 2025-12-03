import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if not cwd in sys.path:
    sys.path.append(cwd)
    
import cv2
import glob

path = "color_spaces/2D/"
path2 = "color_spaces/2D2/"

for name in glob.glob(f"{path}*.png"):
    print(name)
    img = cv2.imread(name)
    img = img[70:-40, 90:-100, :]
    cv2.imwrite(f"{path2}{name.split('/')[-1]}", img)
cv2.destroyAllWindows()

# names = ["HSL", "HSV", "Lab", "Luv", "LCh", "YCrCb", "YUV", "XYZ", "oklab", "RGB"]
# for name in names:
#     img = cv2.imread(f"color_spaces/3D/{name}.png")
#     img = img[20:-20, 135:-70, :]
#     cv2.imwrite(f"color_spaces/3D2/{name}.png", img)