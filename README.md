# MAG_ADNet
This repository is the official implementation of MAG-ADNet. 
# <Project Name>  <!-- 例：MAG_ADNet -->

> <基于多模态的AD和MCI亚种诊断>

[Paper](<link or "TBA">) • [Project Page](<optional>) • [License](#license)

---

## ✨ Model Structure

<p align="center">
  <img src="img/model.png" alt="Model Structure" width="1000"/>
</p>

---

## 💡 Primary Contribution
To further enhance diagnostic accuracy and strengthen the effectiveness of intermediate-to-late multimodal fusion, we propose MAG-ADNet (Modality-Adversarial-Gated Network for AD). This is a novel method which not only optimizes the fusion strategy but also achieves efficient extraction of discriminative information.Our key contributions are summarized as follows:
1. The MAG-ADNet architecture achieves accurate diagnosis with dual modalities by leveraging two modality-specific ResNet-50 encoders together with the MDIF module. This design adversarially learns both modality-specific and fused representations.
2. We introduce Gated Multi-Head Self Attention (GMHSA) to enhance information flow across attention heads and strengthen cross-modal interactions.
3. We adopt DyT to enhance representation learning with a learnable, smoothly adjustable activation curve, improving robustness to distribution shifts and preserving fine-grained multimodal cues.
4. Extensive experiments on both the AD vs CN and LMCI vs EMCI datasets demonstrate superior performance of MAG-ADNet, achieving state-of-the-art results on most metrics.
---

## 🧠 Interpretability

<p align="center">
  <img src="img/interpre.png" alt="Model interpretability visualization" width="1000"/>
</p>

<p align="center">
  <em>Example: Occlusion heatmap showing the key brain regions the model focuses on in MRI / PET.</em>
</p>

---

## 📂 Dataset Structure
We employed the baseline FDG-PET and T1-weighted MRI data, preprocessed and provided by the [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/)
 database.
The dataset should be organized in the following format:

```text
├── mri                             # MRI images
│   ├── train
│   │   ├── AD    # Alzheimer's Disease
│   │   ├── MCI   # Mild Cognitive Impairment
│   │   ├── EMCI  # Early MCI
│   │   └── LMCI  # Late MCI
│   └── test
│       ├── AD
│       ├── MCI
│       ├── EMCI
│       └── LMCI
│
├── pet                             # PET images
│   ├── train
│   │   ├── AD
│   │   ├── MCI
│   │   ├── EMCI
│   │   └── LMCI
│   └── test
│       ├── AD
│       ├── MCI
│       ├── EMCI
│       └── LMCI
│


- `mri/` : contains structural MRI scans in NIfTI format (`.nii.gz`).  
- `pet/` : contains corresponding PET scans in NIfTI format (`.nii.gz`).  
```
---

## 📦 Environment
- Python <3.12.4>
- PyTorch >= <2.4.0>

"Quickstart: Create an Environment (Example)"：
```bash
conda create -n <env_name> python=3.9 -y
conda activate <env_name>
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Train & Test

To train and evaluate the model, simply run:

```bash
python main.py
```

---

## 📊 Results
<h2 style="color:blue;">**Comparative Experiment**</h2>
<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="6" style="text-align:center;">AD vs CN</th>
    <th colspan="6" style="text-align:center;">LMCI vs EMCI</th>
  </tr>
  <tr>
    <th>Acc</th>
    <th>F1</th>
    <th>Kappa</th>
    <th>Sen</th>
    <th>Prec</th>
    <th>Mcc</th>
    <th>Acc</th>
    <th>F1</th>
    <th>Kappa</th>
    <th>Sen</th>
    <th>Prec</th>
    <th>Mcc</th>
  </tr>
  <tr>
    <td>Miccai-Fusion</td>
    <td>85.2</td><td>84.7</td><td>83.9</td><td>84.3</td><td>86.0</td><td>0.89</td>
    <td>78.4</td><td>77.9</td><td>76.5</td><td>77.2</td><td>79.0</td><td>0.82</td>
  </tr>
  <tr>
    <td>ADViT</td>
    <td>89.5</td><td>90.1</td><td>88.6</td><td>89.3</td><td>90.4</td><td>0.92</td>
    <td>83.6</td><td>84.1</td><td>82.7</td><td>83.4</td><td>84.5</td><td>0.87</td>
  </tr>
  <tr>
    <td>Transmf_AD</td>
    <td>87.8</td><td>87.2</td><td>86.1</td><td>86.6</td><td>88.0</td><td>0.90</td>
    <td>80.2</td><td>81.0</td><td>79.4</td><td>80.2</td><td>81.6</td><td>0.84</td>
  </tr>
  <tr>
    <td>MENet</td>
    <td>90.3</td><td>90.7</td><td>89.1</td><td>89.9</td><td>91.0</td><td>0.93</td>
    <td>85.0</td><td>85.6</td><td>84.3</td><td>84.9</td><td>85.8</td><td>0.88</td>
  </tr>
  <tr>
    <td>Diamond</td>
    <td>91.1</td><td>91.5</td><td>90.4</td><td>90.9</td><td>91.8</td><td>0.94</td>
    <td>86.4</td><td>86.9</td><td>85.7</td><td>86.3</td><td>87.2</td><td>0.89</td>
  </tr>
  <tr>
    <td><b>Ours</b></td>
    <td><b>92.7</b></td><td><b>92.3</b></td><td><b>91.8</b></td><td><b>92.0</b></td><td><b>93.1</b></td><td><b>0.95</b></td>
    <td><b>88.9</b></td><td><b>89.2</b></td><td><b>88.1</b></td><td><b>88.6</b></td><td><b>89.7</b></td><td><b>0.91</b></td>
  </tr>
</table>

---

<h2 style="color:blue;">**Ablation Experiment**</h2>
<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="6" style="text-align:center;">AD vs CN</th>
    <th colspan="6" style="text-align:center;">LMCI vs EMCI</th>
  </tr>
  <tr>
    <th>Acc</th>
    <th>F1</th>
    <th>Kappa</th>
    <th>Sen</th>
    <th>Prec</th>
    <th>Mcc</th>
    <th>Acc</th>
    <th>F1</th>
    <th>Kappa</th>
    <th>Sen</th>
    <th>Prec</th>
    <th>Mcc</th>
  </tr>
  <tr>
    <td>(w/o PET)
    </td><td>76.25</td><td>75.79</td><td>51.65</td><td>69.44</td><td>75.76</td><td>51.81</td>
    <td>77.42</td><td>75.44</td><td>51.67</td><td>58.46</td><td>82.61</td><td>53.55</td>
  </tr>
  <tr>
    <td>(w/o MRI)</td>
    <td>86.25</td><td>85.99</td><td>72.01</td><td>80.56</td><td>87.88</td><td>72.22</td>
    <td>78.71</td><td>77.50</td><td>55.23</td><td>66.15</td><td>79.63</td><td>55.85</td>
  </tr>
  <tr>
    <td>(w/o MDIF)</td>
    <td>70.00</td><td>69.70</td><td>39.39</td><td>66.67</td><td>66.67</td><td>39.39</td>
    <td>68.39</td><td>62.81</td><td>29.84</td><td>35.38</td><td>76.67</td><td>34.48</td>
  </tr>
  <tr>
    <td>(w/o DyT)</td>
    <td>85.00</td><td>84.40</td><td>69.10</td><td>72.22</td><td><b>92.90</td><td>70.60</td>
    <td>77.42</td><td>76.13</td><td>52.52</td><td>64.62</td><td>77.78</td><td>53.11</td>
  </tr>
  <tr>
    <td>(w/o GMHSA)</td>
    <td>90.00</td><td>89.84</td><td>79.70</td><td>86.11</td><td>91.18</td><td>79.80</td>
    <td>80.65</td><td>78.69</td><td>58.30</td><td>60.00</td><td><b>90.70</td><td>61.23</td>
  </tr>
  <tr>
    <td><b>Ours</b></td>
    <td><b>92.50</b></td><td><b>92.42</b></td><td><b>84.85</b></td><td><b>91.67</b></td><td>91.67</b></td><td><b>84.85</b></td>
    <td><b>81.29</b></td><td><b>80.09</b></td><td><b>60.48</b></td><td><b>67.69</b></td><td>84.61</b></td><td><b>61.46</b></td>
  </tr>
</table>





