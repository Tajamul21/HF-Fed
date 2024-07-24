<div align="center">

<!-- TITLE -->
# **HF-Fed: Hierarchical based customized Federated Learning Framework for X-Ray Imaging**

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2303.16203-b31b1b.svg)](https://arxiv.org/abs/2303.16203)
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](http://diffusion-classifier.github.io)
</div>

This is the official implementation of the MICCAI 2024 workshop paper [HF-Fed: Hierarchical based customized Federated
Learning Framework for X-Ray Imaging](https://arxiv.org/abs/2303.16203) by Tajamul Ashraf and Tisha Madame.
<!-- DESCRIPTION -->
## Abstract

In clinical applications, X-ray technology plays a crucial
role in noninvasive examinations like mammography, providing essential anatomical information about patients. However, the inherent radiation risk associated with X-Ray procedures raises significant concerns. X-Ray reconstruction is crucial in medical imaging for creating detailed visual representations of internal structures, and facilitating diagnosis
and treatment without invasive procedures. Recent advancements in deep
learning (DL) have shown promise in X-ray reconstruction. Nevertheless,
conventional DL methods often necessitate the centralized aggregation of
substantial large datasets for training, following specific scanning protocols. This requirement results in notable domain shifts and privacy issues.
To address these challenges, we introduce the Hierarchical Framework-based Federated Learning method (HF-Fed) for customized X-Ray Imaging. HF-Fed addresses the challenges in X-ray imaging optimization by decomposing the problem into two components: local data adaptation
and holistic X-ray imaging. It employs a hospital-specific hierarchical
framework and a shared common imaging network called the Network of
Networks (NoN) for these tasks. The emphasis of the NoN is on acquiring
stable features from a variety of data distributions. A hierarchical hypernetwork extracts domain-specific hyperparameters, conditioning the NoN
for customized X-ray reconstruction. Experimental results demonstrate
HF-Fed’s competitive performance, offering a promising solution for enhancing X-Ray imaging without the need for data sharing. This study
significantly contributes to the evolving body of literature on the potential advantages of federated learning in the healthcare sector. It offers
valuable insights for policymakers and healthcare providers holistically. 

## Dataset 

The dataset includes the following files and directories:

```plaintext
RSNA-Breast-Cancer-Detection-Dataset/
├── test_images/
│   ├── image_001.dcm
│   ├── image_002.dcm
│   └── ...
├── train_images/
│   ├── image_001.dcm
│   ├── image_002.dcm
│   └── ...
├── sample_submission.csv
├── test.csv
└── train.csv

```

## Getting Started

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/Tajamul21/HF-Fed.git
    cd HF-Fed
    ```

2. **Download the Dataset**:
    - Visit the [Kaggle competition page](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/)
    - Download the dataset files
    - Extract the dataset files into the appropriate directories


#### Requirements

Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. If you wanna try our method, please first install the necessary packages as follows:

```
pip install requirements.txt
```

Our implementation is based on [CTLib](https://github.com/xiawj-hub/CTLIB) in simulating data and training IR-based methods. If you are interested in data simulation and IR-based networks, we recommend installing it. Furthermore, HF-Fed can be easily integrated into transformer-based methods with minor modifications.

#### Acknowledgments
Special thanks to Prof. Aditeshwar Seth for his support and guidance!

#### Contact
If you have any questions or suggestions about our work, please get in touch with me [Email](mailto:tajamul21.ashraf@gmail.com).



## Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{li2023diffusion,
      title={Your Diffusion Model is Secretly a Zero-Shot Classifier}, 
      author={Alexander C. Li and Mihir Prabhudesai and Shivam Duggal and Ellis Brown and Deepak Pathak},
      year={2023},
      eprint={2303.16203},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


