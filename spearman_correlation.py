from scipy.stats import spearmanr

# SVM data
svm_data_1 = {
    "OKLab": 0.8699,
    "YUV": 0.8459,
    "YCrCb": 0.8459,
    "Lab": 0.8219,
    "Luv": 0.8185,
    "LCh": 0.7979,
    "HSL": 0.7808,
    "HSV": 0.7808,
    "RGB": 0.5411,
    "XYZ": 0.4726,
}

svm_data_2 = {
    "LCh": 0.9932,
    "HSL": 0.9932,
    "YUV": 0.9932,
    "HSV": 0.9897,
    "YCrCb": 0.9897,
    "Luv": 0.9795,
    "Lab": 0.9760,
    "OKLab": 0.9658,
    "XYZ": 0.9486,
    "RGB": 0.9075,
}

svm_data_3 = {
    "HSV": 1.0000,
    "HSL": 1.0000,
    "LCh": 1.0000,
    "YUV": 1.0000,
    "YCrCb": 1.0000,
    "Lab": 1.0000,
    "RGB": 1.0000,
    "OKLab": 1.0000,
    "XYZ": 1.0000,
    "Luv": 1.0000,
}

# Shap data
shap_data_1 = {
    "OKLab": 1711.8754,
    "Lab": 1647.6072,
    "YCrCb": 1452.9532,
    "YUV": 1440.9366,
    "Luv": 1403.8632,
    "LCh": 1222.6041,
    "HSL": 1160.2382,
    "HSV": 1146.6371,
    "RGB": 1086.7613,
    "XYZ": 1035.0892,
}

shap_data_2 = {
    "LCh": 1729.1759,
    "HSV": 1653.4905,
    "HSL": 1626.9923,
    "YUV": 1626.9923,
    "OKLab": 1545.9993,
    "Luv": 1479.0086,
    "Lab": 1452.5369,
    "YCrCb": 1405.0739,
    "XYZ": 1248.1699,
    "RGB": 1114.7241,
}

shap_data_3 = {
    "HSV": 1775.5676,
    "HSL": 1655.1400,
    "LCh": 1605.2955,
    "YUV": 1418.4891,
    "YCrCb": 1412.3514,
    "Lab": 1311.5821,
    "RGB": 1273.8382,
    "OKLab": 1273.2712,
    "XYZ": 1265.4966,
    "Luv": 1239.8212,
}

svm_data_1 = list(dict(sorted(svm_data_1.items())).values())
svm_data_2 = list(dict(sorted(svm_data_2.items())).values())
svm_data_3 = list(dict(sorted(svm_data_3.items())).values())

shap_data_1 = list(dict(sorted(shap_data_1.items())).values())
shap_data_2 = list(dict(sorted(shap_data_2.items())).values())
shap_data_3 = list(dict(sorted(shap_data_3.items())).values())

corr_1, p_value_1 = spearmanr(svm_data_1, shap_data_1)
corr_2, p_value_2 = spearmanr(svm_data_2, shap_data_2)
# corr_3, p_value_3 = spearmanr(svm_data_3, shap_data_3)

print(f"Correlation: {corr_1}, p-value: {p_value_1}")
print(f"Correlation: {corr_2}, p-value: {p_value_2}")
