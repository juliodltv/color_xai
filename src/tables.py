import pandas as pd

def create_accuracy_table():
    data = {
        "HSV_H": 0.9956,
        "HSV_HS": 1.0,
        
        "HSL_H": 0.9956,
        "HSL_HS": 0.9956,
        
        "LCh_L": 0.9824,
        "LCh_LC": 0.99556,
        
        "Lab_a": 0.7489,
        "Lab_b": 0.8678,
        "Lab_ab": 1.0,
        
        "YCrCb_Cr": 0.7048,
        "YCrCb_Cb": 0.8634,
        "YCrCb_CrCb": 0.9912,
        
        "Luv_u": 0.7489,
        "Luv_v": 0.8414,
        "Luv_uv": 0.9912,
        
        "YUV_U": 0.8590,
        "YUV_V": 0.7224,
        "YUV_UV": 0.9912,
    }
    
    df = pd.DataFrame(data.items(), columns=["Channels", "Accuracy"])
    print(df.head(20))
    print(df.to_latex(index=False, float_format="%.4f"))
    
def create_range_table():
    data = {
        "Red": [0.0, 0.0333],
        "Orange": [0.0334, 0.1313],
        "Yellow": [0.1314, 0.2960],
        "Green": [0.2961, 0.3411],
        "Cyan": [0.3412, 0.5745],
        "Blue": [0.5746, 0.7706],
        "Magenta": [0.7707, 1.0],
    }
    
    df = pd.DataFrame(data.items(), columns=["Color", "Range"])
    print(df.head(20))
    print(df.to_latex(index=False, float_format="%.4f"))
    
# create_accuracy_table()
create_range_table()