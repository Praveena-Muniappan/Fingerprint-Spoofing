import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, matthews_corrcoef, 
                             fowlkes_mallows_score, hamming_loss, jaccard_score,
                             roc_curve, roc_auc_score)
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Config ----------

REAL_IMAGES_DIR = r"C:\Users\Sai Raman\OneDrive\Desktop\All Semesters\semester 5\Deep Learning\DL assigment - FingerPrint Spoofing\socofing\versions\2\SOCOFing\SOCOfing_Balanced\SOCOfing_Balanced\Real"
ALTERED_IMAGES_DIRS = [
    r"C:\Users\Sai Raman\OneDrive\Desktop\All Semesters\semester 5\Deep Learning\DL assigment - FingerPrint Spoofing\socofing\versions\2\SOCOFing\SOCOfing_Balanced\SOCOfing_Balanced\Altered\Altered-Easy", 
    r"C:\Users\Sai Raman\OneDrive\Desktop\All Semesters\semester 5\Deep Learning\DL assigment - FingerPrint Spoofing\socofing\versions\2\SOCOFing\SOCOfing_Balanced\SOCOfing_Balanced\Altered\Altered-Medium",
    r"C:\Users\Sai Raman\OneDrive\Desktop\All Semesters\semester 5\Deep Learning\DL assigment - FingerPrint Spoofing\socofing\versions\2\SOCOFing\SOCOfing_Balanced\SOCOfing_Balanced\Altered\Altered-Hard"
]

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- Load Models ----------

def load_models():
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    resnet.to(device)

    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.classifier = torch.nn.Identity()
    mobilenet.eval()
    mobilenet.to(device)

    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    vit.head = torch.nn.Identity()
    vit.eval()
    vit.to(device)

    return resnet, mobilenet, vit

resnet, mobilenet, vit = load_models()

# ---------- Feature Extraction ----------

def extract_features(model, img_path):
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    return features.cpu().numpy().flatten()

# ---------- Collect Dataset ----------

def collect_dataset(real_dir, altered_dirs):
    dataset = []
    for root, _, files in os.walk(real_dir):
        for f in files:
            if f.lower().endswith('.bmp'):
                dataset.append((os.path.join(root, f), 0))  # real = 0
    for altered_dir in altered_dirs:
        for root, _, files in os.walk(altered_dir):
            for f in files:
                if f.lower().endswith('.bmp'):
                    dataset.append((os.path.join(root, f), 1))  # altered = 1
    return dataset

dataset = collect_dataset(REAL_IMAGES_DIR, ALTERED_IMAGES_DIRS)
print(f"Total images collected: {len(dataset)}")

# ---------- Extract Features ----------

features_resnet, features_mobilenet, features_vit = [], [], []
labels = []

print("Extracting features from images...")
for img_path, label in tqdm(dataset):
    features_resnet.append(extract_features(resnet, img_path))
    features_mobilenet.append(extract_features(mobilenet, img_path))
    features_vit.append(extract_features(vit, img_path))
    labels.append(label)

X_resnet = np.array(features_resnet)
X_mobilenet = np.array(features_mobilenet)
X_vit = np.array(features_vit)
y = np.array(labels)

# ---------- Evaluation Helpers ----------

def evaluate_model(y_true, y_pred, clf_name):
    print(f"\n🔎 Extra Evaluation Metrics for {clf_name}:")
    print(f"MCC (Matthews Corr. Coef): {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"FMI (Fowlkes-Mallows Index): {fowlkes_mallows_score(y_true, y_pred):.4f}")
    print(f"Hamming Loss: {hamming_loss(y_true, y_pred):.4f}")
    print(f"Jaccard Score: {jaccard_score(y_true, y_pred):.4f}")

def plot_roc_curve(y_true, y_proba, clf_name, feature_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {clf_name} ({feature_name} Features)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ---------- Train & Evaluate ----------

def train_and_eval(X, y, feature_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n===== Results for {feature_name} Features =====")

    # Decision Tree
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    y_proba_dt = dt_clf.predict_proba(X_test)[:, 1]
    print("Decision Tree Classifier:")
    print(classification_report(y_test, y_pred_dt))
    evaluate_model(y_test, y_pred_dt, "Decision Tree")
    plot_roc_curve(y_test, y_proba_dt, "Decision Tree", feature_name)

    # Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)
    y_proba_lr = lr_clf.predict_proba(X_test)[:, 1]
    print("Logistic Regression Classifier:")
    print(classification_report(y_test, y_pred_lr))
    evaluate_model(y_test, y_pred_lr, "Logistic Regression")
    plot_roc_curve(y_test, y_proba_lr, "Logistic Regression", feature_name)

# ---------- Run training & evaluation ----------
train_and_eval(X_resnet, y, "ResNet50")
train_and_eval(X_mobilenet, y, "MobileNetV2")
train_and_eval(X_vit, y, "ViT_Base")
