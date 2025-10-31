from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Any, List, Dict, Optional
import logging
import asyncio
# Example: Simple model for demonstration (replace with your real model)
class SimpleCyberModel(nn.Module):
    def __init__(self, input_dim=10, num_classes=2) -> Any:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x) -> Any:
        return self.net(x)

# Instantiate and load model (replace with your trained model)
input_dim = 10
num_classes = 2
model = SimpleCyberModel(input_dim=input_dim, num_classes=num_classes)
model.eval()

def sanitize_metric(value) -> Any:
    if isinstance(value, np.ndarray):
        value = value.astype(float)
        value[~np.isfinite(value)] = 0.0
        return value
    if not np.isfinite(value):
        return 0.0
    return value

def predict(features) -> Any:
    # features: list of floats
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred = int(np.argmax(probs))
    probs = sanitize_metric(probs)
    return {f"Class {i}": float(p) for i, p in enumerate(probs)}, pred

def batch_predict(batch) -> Any:
    # batch: list of lists
    x = torch.tensor(batch, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    probs = sanitize_metric(probs)
    return preds, probs

def plot_confusion_matrix(y_true, y_pred) -> Any:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_score) -> Any:
    if y_score.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
    return None

def plot_pr_curve(y_true, y_score) -> Any:
    if y_score.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label="PR curve")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        plt.tight_layout()
        return fig
    return None

def batch_metrics(batch, labels) -> Any:
    preds, probs = batch_predict(batch)
    y_true = np.array(labels)
    y_pred = preds
    # Sanitize
    y_true = sanitize_metric(y_true)
    y_pred = sanitize_metric(y_pred)
    # Confusion matrix
    cm_fig = plot_confusion_matrix(y_true, y_pred)
    # ROC curve
    roc_fig = plot_roc_curve(y_true, probs)
    # PR curve
    pr_fig = plot_pr_curve(y_true, probs)
    return cm_fig, roc_fig, pr_fig

# Gradio interface for single prediction
single_input = [gr.Number(label=f"Feature {i+1}") for i in range(input_dim)]
single_output = [gr.Label(num_top_classes=num_classes, label="Class Probabilities"), gr.Number(label="Predicted Class")]

single_demo = gr.Interface(
    fn=predict,
    inputs=single_input,
    outputs=single_output,
    title="Cybersecurity Model Inference (Single Input)",
    description="Enter feature values to get class probabilities and prediction. Handles NaN/Inf gracefully."
)

# Gradio interface for batch prediction and metrics
batch_input = [
    gr.Dataframe(headers=[f"Feature {i+1}" for i in range(input_dim)], label="Batch Features", type="numpy"),
    gr.Dataframe(headers=["Label"], label="True Labels", type="numpy")
]
batch_output = [
    gr.Plot(label="Confusion Matrix"),
    gr.Plot(label="ROC Curve"),
    gr.Plot(label="Precision-Recall Curve")
]

batch_demo = gr.Interface(
    fn=batch_metrics,
    inputs=batch_input,
    outputs=batch_output,
    title="Cybersecurity Model Batch Metrics",
    description="Upload a batch of features and true labels to visualize confusion matrix, ROC, and PR curves."
)

demo = gr.TabbedInterface(
    [single_demo, batch_demo],
    tab_names=["Single Inference", "Batch Metrics"]
)

match __name__:
    case "__main__":
    demo.launch() 