# GLASS
- https://arxiv.org/abs/2407.09359

GLASS is an anomaly detection method that learns only normal images and then generates gradient-ascent–controlled pseudo anomalies at both the feature level (Global) and the image level (Local), enabling the model to learn and detect even subtle, hard-to-distinguish defects.

## Scripts
 - shell/download.sh # Download Dataset and Results(.pth)

## Runtime Environment
- Google Colab 
  - GPU RAM 4.4GB
  - System RAM 9.6GB
  - Disk 40.9 GB

## How to Train and Test
 Step.1 !git clone https://github.com/masa8/GLASS.git   
 Step.2 !wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz   
 Step.3 !tar -xf dtd-r1.0.1.tar.gz  
 Step.4 !pip install -U "numpy<2" "imgaug==0.4.0".  
 Step.5 Put the dataset as follows:  
         /content/GLASS/datasets/mvtec/screw/test/good 
         /content/GLASS/datasets/mvtec/screw/test/not-good  
         /content/GLASS/datasets/mvtec/screw/train/good  
         /content/GLASS/datasets/mvtec/screw/train/not-good  
 Step.6 !cd GLASS/shell && bash run-mvtec.sh   

## Results
 - Model: GLASS/results/models/backbone_0/ckpt_best_*.pth
 - ROC for test: GLASS/results/roc_curves/mvtec_screw_image_roc.png
 - Test Results: GLASS/results/test_results/mvtec_screw/all_test_results.csv

## How to run test-only inference
 Step.1 Put ckpt_best_*.pth in GLASS/results/models/backbone_0/mvtec_screw/
 Step.2 Update -test option
  - When the `-test` option in run-mvtec.sh is set to a value other than `ckpt`, the script skips training and runs inference on the test set.

 
## License
The code and dataset in this repository are licensed under the [MIT license](https://github.com/cqylunlun/GLASS?tab=MIT-1-ov-file/).
