import sys
import os
import time
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import colorspacious
import cv2

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- Data Loading and Preprocessing (Copied/Adapted from src/main.py) ---
class DataAugmentation:
    def augment_data_spaces(self, df):
        # ... (Same implementation as in src/main.py) ...
        # For brevity, I'll implement the core logic needed
        df["LCh"] = df["RGB"].apply(lambda x: colorspacious.cspace_convert(x, "sRGB255", "CIELCh")/np.array([100, 100, 360]))
        df["HSV"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2HSV)[0][0]/np.array([180, 255, 255]))
        df["HSL"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2HLS)[0][0]/np.array([180, 255, 255]))
        df["YCrCb"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2YCrCb)[0][0]/255)
        df["XYZ"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2XYZ)[0][0]/255)
        df["Lab"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2Lab)[0][0]/255)
        df["Luv"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2Luv)[0][0]/255)
        df["YUV"] = df["RGB"].apply(lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2YUV)[0][0]/255)
        df["RGB"] = df["RGB"].apply(lambda x: np.uint8(x)/255)
        return df

    def separate_channels(self, df):
        # ... (Same implementation as in src/main.py) ...
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
        
        df = df.drop(columns=["RGB", "LCh","HSV", "HSL", "YCrCb", "XYZ", "Lab", "Luv", "YUV"])
        return df

def load_data():
    # Try loading from pickle, if not found, create dummy data for testing structure
    try:
        df = pd.read_pickle("real_data_regras/video.pkl")
    except FileNotFoundError:
        print("Data file not found, please ensure 'real_data_regras/video.pkl' exists.")
        sys.exit(1)
        
    da = DataAugmentation()
    df = da.augment_data_spaces(df)
    df = da.separate_channels(df)
    
    color_to_int = {
        "Blue": 0, "Yellow": 1, "Cyan": 2, "Green": 3, "Red": 4,
        "Magenta": 5, "Orange": 6, "White": 7, "Black": 8,
    }
    y = df["Color"].map(color_to_int)
    X = df.drop(columns=["Color"])
    
    # Handle potential NaN values if any map failed (though it shouldn't)
    if y.isna().any():
        print("Warning: NaN labels found, dropping rows.")
        X = X[~y.isna()]
        y = y[~y.isna()]
        
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Models ---

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(27, 15) # Updated to 15
        self.act = nn.ReLU()
        self.output = nn.Linear(15, 9)
        
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
    
    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(X_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()

def train_torch(X_train, y_train):
    model = TorchModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_t = torch.tensor(y_train.values, dtype=torch.int64)
    
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    
    for epoch in range(500): # 500 epochs
        for X_b, y_b in loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel="rbf", decision_function_shape="ovr", C=100.0, gamma=1.0, probability=True)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=35) # Updated to 35
    model.fit(X_train, y_train)
    return model

def train_lgb(X_train, y_train):
    d_train = lgb.Dataset(X_train, label=y_train)
    params = {
        "num_threads": 8, "max_bin": 512, "learning_rate": 0.05,
        "objective": "multiclass", "metric": "multi_logloss",
        "num_leaves": 10, "verbose": -1, "min_data": 100,
        "num_class": 9, "max_depth": 2,
    }
    model = lgb.train(params, d_train, 1000)
    return model

class LGBWrapper:
    def __init__(self, model):
        self.model = model
    def predict_proba(self, X):
        return self.model.predict(X)
    def predict(self, X):
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

# --- Main Comparison Logic ---

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    def to_numpy(x):
        if hasattr(x, 'values'):
            return x.values
        return x

    models = [
        ("SVM", train_svm, lambda m, x: m.predict(to_numpy(x)), lambda m, x: m.predict_proba(to_numpy(x))),
        ("KNN", train_knn, lambda m, x: m.predict(to_numpy(x)), lambda m, x: m.predict_proba(to_numpy(x))),
        ("Neural Network", train_torch, 
         lambda m, x: np.argmax(m.predict_proba(to_numpy(x)), axis=1), 
         lambda m, x: m.predict_proba(to_numpy(x))),
        ("LightGBM", train_lgb, 
         lambda m, x: np.argmax(m.predict(x), axis=1), 
         lambda m, x: m.predict(x))
    ]
    
    results = []
    
    # Use a small background dataset for SHAP to speed up
    background_summary = shap.kmeans(X_train, 10) 
    
    # Use a small test set for SHAP calculation
    X_shap = X_test.iloc[:50] 
    
    for name, train_func, pred_func, proba_func in models:
        print(f"\nProcessing {name}...")
        
        # Train
        start_time = time.time()
        model = train_func(X_train, y_train)
        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.2f}s")
        
        # Accuracy
        y_pred = pred_func(model, X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")
        
        # SHAP
        print("  Calculating SHAP values...")
        start_shap = time.time()
        explainer = shap.KernelExplainer(lambda x: proba_func(model, x), background_summary)
        # Suppress warnings
        with np.errstate(divide='ignore', invalid='ignore'):
             shap_values = explainer.shap_values(X_shap, nsamples=100, silent=True)
        shap_time = time.time() - start_shap
        
        # Metrics
        # shap_values is a list of arrays (one for each class)
        # We aggregate absolute importance across all classes
        total_importance = np.sum([np.abs(s) for s in shap_values], axis=0) # shape (n_samples, n_features)
        mean_importance = np.mean(total_importance)
        
        # Sparsity: Percentage of features with importance < 1% of max importance per sample
        sparsity_scores = []
        for sample_imp in total_importance:
            threshold = 0.01 * np.max(sample_imp)
            sparse_count = np.sum(sample_imp < threshold)
            sparsity_scores.append(sparse_count / len(sample_imp))
        avg_sparsity = np.mean(sparsity_scores)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Train Time (s)": train_time,
            "SHAP Time (s)": shap_time,
            "Mean |SHAP|": mean_importance,
            "Sparsity": avg_sparsity
        })

    # Output Table
    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(df_res.to_string(index=False, float_format="%.4f"))
    
    print("\n" + "="*80)
    print("LaTeX Table")
    print("="*80)
    print(df_res.to_latex(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()
