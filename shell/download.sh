#!/usr/bin/env bash
set -e

# download DTD dataset (for texture augmentation)
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xf dtd-r1.0.1.tar.gz

# install imgaug (for data augmentation)
pip install -U "numpy<2" "imgaug==0.4.0"

# download screw dataset (for anomaly detection)
# ===== 設定 =====
FOLDER_ID="1a0onTe-bOrO6bzGZPwlUNZAY-_mFE1fJ"
OUT_DIR="screw-shared-data"
# =================

echo "Installing gdown..."
pip -q install gdown

echo "Downloading folder from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$OUT_DIR"

echo "Saved to directory: $OUT_DIR"

unzip -q screw-shared-data/screw-datasets.zip
# screwフォルダ自体を移動（mvtec/screw/になるように）
mv screw GLASS/datasets/mvtec/
