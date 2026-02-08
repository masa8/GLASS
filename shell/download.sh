#!/usr/bin/env bash
set -e


#wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
#tar -xf dtd-r1.0.1.tar.gz
#pip install -U "numpy<2" "imgaug==0.4.0"

# ===== 設定 =====
FOLDER_ID="1a0onTe-bOrO6bzGZPwlUNZAY-_mFE1fJ"
OUT_DIR="screw-shared-data"
# =================

echo "Installing gdown..."
pip -q install gdown

echo "Downloading folder from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$OUT_DIR"

echo "Done."
echo "Saved to directory: $OUT_DIR"


unzip screw-shared-data/screw-datasets.zip
mv screw/test/ GLASS/datasets/mvtec/screw/
mv screw/train/ GLASS/datasets/mvtec/screw/