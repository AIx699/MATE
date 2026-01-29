# MATE: A Motion-Aware Dynamic Attention for Efficient Anomaly Detection

## Abstract

Efficient and accurate video anomaly detection is critical for real-time applications such as autonomous driving, large-scale surveillance, and security monitoring. However, many existing approaches rely on implicit motion representations, which often struggle to capture complex, subtle, or rapidly evolving anomalies, limiting both detection accuracy and computational efficiency. To address this gap, we propose **MATE** (**M**otion-**A**ware **A**ttention and **T**emporal-**C**onditioned **C**onvolution **E**ncoder), a lightweight motion-conditioned architecture that explicitly integrates motion magnitude and directional information throughout the network. MATE leverages directional motion maps, motion-gated multi-scale temporal-conditioned convolutions, motion-modulated kernelized linear attention, and motion-aware temporal pooling to focus computation on anomaly-relevant regions while remaining highly parameter-efficient. On the UCF-Crime benchmark under the standard video-level evaluation protocol, MATE-FAST and MATE-BASE achieve ROC-AUC scores of **89.75%** and **91.78%**, respectively, and outperform recent strong baselines while using substantially fewer parameters (FAST ≈ 8.5k, BASE ≈ 40k). Extensive ablation studies, synthetic signal-to-noise ratio experiments, and per-fold cross-validation demonstrate that explicit motion conditioning improves temporal feature discriminability and embedding geometry under weak supervision. These results position MATE as a practical and effective solution for low-resource and low-latency video anomaly detection. Code and pretrained models are publicly available at https://github.com/AIx699/MATE.

## Architecture

<img width="1122" height="422" alt="Arch2" src="https://github.com/user-attachments/assets/09928364-bce2-472c-aa59-c18c794af77e" />


## Dataset

- **UCF-Crime** dataset information: https://www.crcv.ucf.edu/projects/real-world/
- UCF-Crime 10-crop I3D features: https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc
- Extracted X3D features UCF-Crime: https://drive.google.com/file/d/1TUkZN4zWWeHpFkr24FvWNIuqOIHy96EJ/view?usp=sharing
- **XD-Violence** dataset information and 5-crop I3D features: https://roc-ng.github.io/XD-Violence/

## Environment

```bash
    pip install -r requirements.txt
```

## Train

```bash
python main.py \
  --train_list train.txt \
  --val_list   test.txt \
  --batch_size 16 \
  --lr 0.002 \
  --max_epoch 250 \
  --model_arch fast \
  --model_name motion_fast
```

## Inference 

```bash
python test.py --model_arch fast
```
## Result on UCF CRIME 

| Method                                     |  Feature            | AUC-ROC |
|-------------------------------------------:|------------------:  |--------:|
| BN-WVAD                                    | I3D                 | 87.24   |
| STEAD                                      | X3D                 | 91.34   |
| MATE                                       | I3D                 | 89.05   |
| MATE                                       | X3D                 | 91.77   |

## Thanks to

- BN-WVAD 2023  https://github.com/cool-xuan/BN-WVAD
- STEAD 2025 https://github.com/agao8/STEAD
