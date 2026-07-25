import os
import time
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage import io, filters
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
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

def extract_edge_features(gray):
    edges_canny = cv2.Canny(gray, 100, 200)
    sobelx = filters.sobel_h(gray)
    sobely = filters.sobel_v(gray)
    sobel = np.hypot(sobelx, sobely)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.array([
        edges_canny.mean(), edges_canny.var(),
        sobel.mean(), sobel.var(),
        laplacian.mean(), laplacian.var()
    ])

def extract_all_features(img_path):
    img = io.imread(img_path)

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.astype(np.uint8)

    gray = cv2.resize(gray, (128, 128))

    lbp = extract_lbp_features(gray)
    gabor = extract_gabor_features(gray)
    glcm = extract_glcm_features(gray)
    hog_features = extract_hog_features(gray)
    edges = extract_edge_features(gray)

    return np.hstack([lbp, gabor, glcm, hog_features, edges])

# ---------- Extract Features ----------
features, labels = [], []
print("Extracting handcrafted features...")
for i, (img_path, label) in enumerate(tqdm(dataset)):
    try:
        feats = extract_all_features(img_path)
        features.append(feats)
        labels.append(label)
        if i % 500 == 0 and i > 0:
            print(f"[INFO] Processed {i}/{len(dataset)} images...")
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")

X = np.array(features)
y = np.array(labels)
print(f"Feature matrix shape: {X.shape}")

# ---------- Train & Evaluate ----------
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Print classification metrics
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    # ROC curve for binary classification
    if len(np.unique(y_test)) == 2:  # Binary classification
        y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_score)
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\n===== Results for Handcrafted Features =====")
    evaluate_model(DecisionTreeClassifier(random_state=42), X_train, X_test, y_train, y_test, "Decision Tree")
    evaluate_model(LogisticRegression(max_iter=1000, random_state=42), X_train, X_test, y_train, y_test, "Logistic Regression")

train_and_eval(X, y)
