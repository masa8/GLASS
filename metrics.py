from sklearn import metrics
from skimage import measure

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "ap": ap}


def plot_roc_curve(anomaly_prediction_weights, anomaly_ground_truth_labels, save_path, dataset_name=''):
    """
    Plot ROC curve and save as PNG.
    
    Args:
        anomaly_prediction_weights: Predicted anomaly scores
        anomaly_ground_truth_labels: Ground truth labels (0=normal, 1=anomaly)
        save_path: Path to save the PNG file
        dataset_name: Name of the dataset for the title
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {dataset_name}' if dataset_name else 'ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return auroc


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    return {"auroc": auroc, "ap": ap}


def compute_pro(masks, amaps, num_th=200):
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, index=[0])])

    df = df[df["fpr"] < 0.3]
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc
