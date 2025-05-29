import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ✅ Load the true labels & predicted labels from evaluations
true_labels = np.load("datasets/image_dataset/processed_faces/image_labels.npy")  # Replace with correct label file
pred_text_xgb = np.load("backend/ml/results/text_xgb_predictions.npy")  # Replace with actual predictions
pred_text_rf = np.load("backend/ml/results/text_rf_predictions.npy")
pred_image_xgb = np.load("backend/ml/results/image_xgb_predictions.npy")
pred_image_rf = np.load("backend/ml/results/image_rf_predictions.npy")
pred_merged = np.load("backend/ml/results/merged_predictions.npy")  # Replace with merged model predictions

# ✅ Define a function to plot confusion matrices
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"datasets/{model_name}_confusion_matrix.png")  # Save the figure
    plt.show()

# ✅ Generate confusion matrices for all models
plot_confusion_matrix(true_labels, pred_text_xgb, "Text_XGBoost")
plot_confusion_matrix(true_labels, pred_text_rf, "Text_RandomForest")
plot_confusion_matrix(true_labels, pred_image_xgb, "Image_XGBoost")
plot_confusion_matrix(true_labels, pred_image_rf, "Image_RandomForest")
plot_confusion_matrix(true_labels, pred_merged, "Merged_Model")

print("✅ Confusion matrices generated and saved in 'datasets/' folder.")
