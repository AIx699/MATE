# MATE: A Motion-Aware Dynamic Attention and Temporal-Conditioned Convolution Encoder for Efficient Anomaly Detection

This repository implements a motion-conditioned architecture for weakly-supervised video anomaly detection . The model conditions attention and pooling on an explicit MotionMap derived from feature differences and operates on precomputed X3D or I3D features extracted from UCF-Crime and XD-Violence videos.

## Dataset
- UCF-Crime dataset information: https://www.crcv.ucf.edu/projects/real-world/
- UCF-Crime 10-crop I3D features: https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc
- Extracted X3D features: https://drive.google.com/file/d/1TUkZN4zWWeHpFkr24FvWNIuqOIHy96EJ/view?usp=sharing
- XD-Violence dataset information and 5-crop I3D features: https://roc-ng.github.io/XD-Violence/

## Train

```bash
python main.py \
  --train_list train.txt \
  --val_list   test.txt \
  --batch_size 16 \
  --lr 0.002 \
  --max_epoch 250 \
  --exp_name "motion_cond_fast" \
  --model_variant fast
```

## Inference 

```bash
python test.py
```
## Result on UCF CRIME 

| Component Removed                          | # Feature Extractor | AUC-ROC |
|-------------------------------------------:|------------------:  |--------:|
| BN-WVAD                                    | I3D                 | 87.24   |
| STEAD                                      | X3D                 | 91.34   |
| MATE                                       | I3D                 | 89.05   |
| MATE                                       | X3D                 | 91.77   |

## Thanks to

BN-WVAD 2023  https://github.com/cool-xuan/BN-WVAD
STEAD 2025 https://github.com/agao8/STEAD
