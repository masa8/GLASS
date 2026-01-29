#!/usr/bin/env bash
set -e

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
