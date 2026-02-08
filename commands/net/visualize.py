"""特徴量分布の可視化関数"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging

LOGGER = logging.getLogger(__name__)


def visualize_feature_distribution(all_features, center, name, epoch):
    """特徴量分布を可視化してPNGとして保存

    Args:
        all_features: list of torch.Tensor - 全バッチの特徴量リスト
        center: torch.Tensor - 特徴量の中心点
        name: str - カテゴリ名
        epoch: int or str - エポック番号または"test"などの文字列
    """
    # 全特徴量を結合
    all_features = torch.cat(all_features, dim=0).numpy()  # (N, feature_dim)

    # PCAで2次元に削減
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)

    # 中心点も2次元に投影
    center_np = center.mean(dim=0).cpu().numpy().reshape(1, -1)
    center_2d = pca.transform(center_np)

    # 中心からの距離を計算
    center_full = center.mean(dim=0).cpu().numpy()
    distances = np.linalg.norm(all_features - center_full, axis=1)

    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左: PCA散布図
    scatter = ax1.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=distances,
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    ax1.scatter(
        center_2d[0, 0],
        center_2d[0, 1],
        c="red",
        marker="x",
        s=200,
        linewidths=3,
        label="Center",
    )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax1.set_title(f"Feature Distribution (PCA) - Epoch {epoch}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Distance from Center")

    # 右: 距離のヒストグラム
    ax2.hist(distances, bins=50, alpha=0.7, edgecolor="black")
    ax2.axvline(
        distances.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {distances.mean():.2f}",
    )
    ax2.axvline(
        distances.std(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Std: {distances.std():.2f}",
    )
    ax2.set_xlabel("Distance from Center")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Distance Distribution - Epoch {epoch}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 統計情報を追加
    stats_text = f"N={len(distances)}\nMean={distances.mean():.2f}\n"
    stats_text += f"Std={distances.std():.2f}\nMin={distances.min():.2f}\n"
    stats_text += f"Max={distances.max():.2f}"
    ax2.text(
        0.98,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
    )

    plt.tight_layout()

    # 保存
    save_dir = f"./results/feature_distribution/{name}"
    os.makedirs(save_dir, exist_ok=True)
    # epochが数値の場合は4桁でゼロパディング、文字列の場合はそのまま使用
    epoch_str = f"{epoch:04d}" if isinstance(epoch, int) else str(epoch)
    save_path = os.path.join(save_dir, f"epoch_{epoch_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    LOGGER.info(f"Feature distribution saved: {save_path}")


def visualize_tsne_distribution(all_features, center, name, epoch):
    """t-SNEによる特徴量分布の可視化

    Args:
        all_features: list of torch.Tensor - 全バッチの特徴量リスト
        center: torch.Tensor - 特徴量の中心点
        name: str - カテゴリ名
        epoch: int or str - エポック番号または"test"などの文字列
    """
    from sklearn.manifold import TSNE

    # 全特徴量を結合
    all_features = torch.cat(all_features, dim=0).numpy()

    # t-SNEで2次元に削減（時間がかかるので警告）
    LOGGER.info(f"Computing t-SNE for epoch {epoch}... (this may take a while)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)

    # 中心からの距離を計算
    center_full = center.mean(dim=0).cpu().numpy()
    distances = np.linalg.norm(all_features - center_full, axis=1)

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=distances,
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title(f"Feature Distribution (t-SNE) - Epoch {epoch}")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Distance from Center")

    plt.tight_layout()

    # 保存
    save_dir = f"./results/feature_distribution_tsne/{name}"
    os.makedirs(save_dir, exist_ok=True)
    # epochが数値の場合は4桁でゼロパディング、文字列の場合はそのまま使用
    epoch_str = f"{epoch:04d}" if isinstance(epoch, int) else str(epoch)
    save_path = os.path.join(save_dir, f"epoch_{epoch_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    LOGGER.info(f"t-SNE distribution saved: {save_path}")


def visualize_anomaly_score_histogram(scores, labels_gt, name):
    """異常スコアのヒストグラムを可視化（正常と異常を別々の線で表示）

    Args:
        scores: np.ndarray - 異常スコア (N,)
        labels_gt: np.ndarray - 正解ラベル (N,) 0=正常, 1=異常
        name: str - カテゴリ名
    """
    # 正常と異常に分離
    normal_scores = scores[labels_gt == 0]
    anomaly_scores = scores[labels_gt == 1]

    # 可視化
    fig, ax = plt.subplots(figsize=(12, 6))

    # ヒストグラムのビン数を自動調整（データ数に応じて）
    bins = min(50, max(20, len(scores) // 10))

    # 正常サンプルのヒストグラム（緑色の線）
    if len(normal_scores) > 0:
        ax.hist(
            normal_scores,
            bins=bins,
            alpha=0.6,
            color="green",
            edgecolor="darkgreen",
            label=f"Normal (n={len(normal_scores)})",
            histtype="step",
            linewidth=2,
        )
        ax.hist(
            normal_scores,
            bins=bins,
            alpha=0.3,
            color="green",
        )

    # 異常サンプルのヒストグラム（赤色の線）
    if len(anomaly_scores) > 0:
        ax.hist(
            anomaly_scores,
            bins=bins,
            alpha=0.6,
            color="red",
            edgecolor="darkred",
            label=f"Anomaly (n={len(anomaly_scores)})",
            histtype="step",
            linewidth=2,
        )
        ax.hist(
            anomaly_scores,
            bins=bins,
            alpha=0.3,
            color="red",
        )

    # 統計情報を追加
    stats_text = ""
    if len(normal_scores) > 0:
        stats_text += "Normal:\n"
        stats_text += f"  Mean: {normal_scores.mean():.4f}\n"
        stats_text += f"  Std: {normal_scores.std():.4f}\n"
        stats_text += f"  Min: {normal_scores.min():.4f}\n"
        stats_text += f"  Max: {normal_scores.max():.4f}\n"

    if len(anomaly_scores) > 0:
        if stats_text:
            stats_text += "\n"
        stats_text += "Anomaly:\n"
        stats_text += f"  Mean: {anomaly_scores.mean():.4f}\n"
        stats_text += f"  Std: {anomaly_scores.std():.4f}\n"
        stats_text += f"  Min: {anomaly_scores.min():.4f}\n"
        stats_text += f"  Max: {anomaly_scores.max():.4f}"

    # 統計情報のテキストボックス
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
        family="monospace",
    )

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Anomaly Score Distribution - {name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    # 保存
    save_dir = f"./results/anomaly_score_histogram/{name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "test_score_histogram.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    LOGGER.info(f"Anomaly score histogram saved: {save_path}")
