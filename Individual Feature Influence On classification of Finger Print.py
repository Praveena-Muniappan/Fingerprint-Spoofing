import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage import io, filters
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, matthews_corrcoef,
    fowlkes_mallows_score, hamming_loss, jaccard_score,
    roc_auc_score, roc_curve
)
from tqdm import tqdm

# ---------- Config ----------
REAL_IMAGES_DIR = r"C:\Users\Sai Raman\OneDrive\Desktop\All Semesters\semester 5\Deep Learning\DL assigment - FingerPrint Spoofing\socofing\versions\2\SOCOFing\SOCOfing_Balanced\SOCOfing_Balanced\Real"
ALTERED_IMAGES_DIRS = [
    r"C:\Users\Sai Raman\OneDrive\Desktop\All Semesters\semester 5\Deep Learning\DL assigment - FingerPrint Spoofing\socofing\versions\2\SOCOFing\SOCOfing_Balanced\SOCOfing_Balanced\Altered\Altered-Easy",
    r"C:\Users\Sai Raman\OneDrive\Desktop\All Semesters\semester 5\Deep Learning\DL assigment - FingerPrint Spoofing\socofing\versions\2\SOCOFing\SOCOfing_Balanced\SOCOfing_Balanced\Altered\Altered-Medium",
    r"C:\Users\Sai Raman\OneDrive\Desktop\All Semesters\semester 5\Deep Learning\DL assigment - FingerPrint Spoofing\socofing\versions\2\SOCOFing\SOCOfing_Balanced\SOCOfing_Balanced\Altered\Altered-Hard"
]

# ---------- Dataset Collection ----------
def collect_dataset(real_dir, altered_dirs):
    dataset = []
    for root, _, files in os.walk(real_dir):
        for f in files:
            if f.lower().endswith('.bmp'):
                dataset.append((os.path.join(root, f), 0))
    for altered_dir in altered_dirs:
        for root, _, files in os.walk(altered_dir):
            for f in files:
                if f.lower().endswith('.bmp'):
                    dataset.append((os.path.join(root, f), 1))
    return dataset

dataset = collect_dataset(REAL_IMAGES_DIR, ALTERED_IMAGES_DIRS)
print(f"Total images collected: {len(dataset)}")

# ---------- Feature Extraction Functions ----------
def extract_lbp_features(gray):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_gabor_features(gray):
    features = []
    num_orientations = 4
    for theta in range(num_orientations):
        theta_val = theta / num_orientations * np.pi
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta_val, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        features.append(filtered.mean())
        features.append(filtered.var())
    return np.array(features)

def extract_glcm_features(gray):
    gray = (gray / 16).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ])

def extract_hog_features(gray):
    features, _ = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=True, block_norm="L2-Hys")
    return features

# --- Edge Features ---
def extract_canny_features(gray):
    edges_canny = cv2.Canny(gray, 100, 200)
    return np.array([edges_canny.mean(), edges_canny.var()])

def extract_sobel_features(gray):
    sobelx = filters.sobel_h(gray)
    sobely = filters.sobel_v(gray)
    sobel = np.hypot(sobelx, sobely)
    return np.array([sobel.mean(), sobel.var()])

def extract_laplacian_features(gray):
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.array([laplacian.mean(), laplacian.var()])

# ---------- Extract Features ----------
features_dict = {
    "LBP": [],
    "Gabor": [],
    "GLCM": [],
    "HOG": [],
    "Canny": [],
    "Sobel": [],
    "Laplacian": []
}
labels = []

print("Extracting handcrafted features...")
for i, (img_path, label) in enumerate(tqdm(dataset)):
    try:
        img = io.imread(img_path)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.astype(np.uint8)

        gray = cv2.resize(gray, (128, 128))

        features_dict["LBP"].append(extract_lbp_features(gray))
        features_dict["Gabor"].append(extract_gabor_features(gray))
        features_dict["GLCM"].append(extract_glcm_features(gray))
        features_dict["HOG"].append(extract_hog_features(gray))
        features_dict["Canny"].append(extract_canny_features(gray))
        features_dict["Sobel"].append(extract_sobel_features(gray))
        features_dict["Laplacian"].append(extract_laplacian_features(gray))

        labels.append(label)

        if i % 500 == 0 and i > 0:
            print(f"[INFO] Processed {i}/{len(dataset)} images...")
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")

y = np.array(labels)
print("Feature extraction completed.")

# ---------- Train & Evaluate for each feature set ----------
def train_and_eval(X, y, feature_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n===== Results using {feature_name} Features =====")

    plt.figure(figsize=(8, 6))

    # --- Decision Tree ---
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    y_prob_dt = dt_clf.predict_proba(X_test)[:, 1]

    print("Decision Tree Classifier:")
    print(classification_report(y_test, y_pred_dt))
    print(f"MCC: {matthews_corrcoef(y_test, y_pred_dt):.4f}")
    print(f"FMI: {fowlkes_mallows_score(y_test, y_pred_dt):.4f}")
    print(f"Hamming Loss: {hamming_loss(y_test, y_pred_dt):.4f}")
    print(f"Jaccard Score: {jaccard_score(y_test, y_pred_dt):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_dt):.4f}")

    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
    plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={roc_auc_score(y_test, y_prob_dt):.2f})")

    # --- Logistic Regression ---
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)
    y_prob_lr = lr_clf.predict_proba(X_test)[:, 1]

    print("Logistic Regression Classifier:")
    print(classification_report(y_test, y_pred_lr))
    print(f"MCC: {matthews_corrcoef(y_test, y_pred_lr):.4f}")
    print(f"FMI: {fowlkes_mallows_score(y_test, y_pred_lr):.4f}")
    print(f"Hamming Loss: {hamming_loss(y_test, y_pred_lr):.4f}")
    print(f"Jaccard Score: {jaccard_score(y_test, y_pred_lr):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={roc_auc_score(y_test, y_prob_lr):.2f})")

    # --- ROC Plot ---
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {feature_name} Features")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# ---------- Run for all features ----------
for feat_name, feat_values in features_dict.items():
    X = np.array(feat_values)
    train_and_eval(X, y, feat_name)
