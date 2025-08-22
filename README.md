# MAG_ADNet
python implementation of MAG_ADNet
# <Project Name>  <!-- 例：MAG_ADNet -->

> <基于多模态的AD和MCI亚种诊断>

[Paper](<link or "TBA">) • [Project Page](<optional>) • [License](#license)

---

## ✨ Model Structure

<p align="center">
  <img src="img/model.png" alt="Model Structure" width="1000"/>
</p>

---

## 💡 Features
- 多模态支持：MRI / PET
- <核心方法/模块>（ResNet-50 、 Modality Feature Discrepancy Identification and Fusion (MDIF) 、 Dynamic Tanh 、 Gated Multi-Head Self Attention (GMHSA)）
- 复现实验脚本与配置

---

## 📂 Datasets

The dataset should be organized in the following structure:
├── mri # MRI images
│ ├── train
│ │ ├── AD # Alzheimer's Disease
│ │ ├── MCI # Mild Cognitive Impairment
│ │ ├── EMCI # Early MCI
│ │ └── LMCI # Late MCI
│ └── test
│ ├── AD
│ ├── MCI
│ ├── EMCI
│ └── LMCI
│
├── pet # PET images
│ ├── train
│ │ ├── AD
│ │ ├── MCI
│ │ ├── EMCI
│ │ └── LMCI
│ └── test
│ ├── AD
│ ├── MCI
│ ├── EMCI
│ └── LMCI
│

- `MRI/` : contains structural MRI scans in NIfTI format (`.nii.gz`).  
- `PET/` : contains corresponding PET scans in NIfTI format (`.nii.gz`).  

---

## 📦 Environment
- Python <3.12.4>
- PyTorch >= <2.4.0>

快速创建环境（示例）：
```bash
conda create -n <env_name> python=3.9 -y
conda activate <env_name>
pip install -r requirements.txt

