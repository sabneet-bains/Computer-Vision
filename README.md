<div align="center"><a name="readme-top"></a>

# 👁️ Computer Vision — Classical × Deep Learning Systems

[![Python](https://img.shields.io/badge/Python-3.9%2B-528ec5?logo=python&logoColor=white&labelColor=0d1117&style=flat)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-d13c3c?logo=opencv&logoColor=white&labelColor=0d1117&style=flat)](https://opencv.org/)
[![TensorFlow/Keras](https://img.shields.io/badge/TensorFlow-Keras-f9a03c?logo=tensorflow&logoColor=white&labelColor=0d1117&style=flat)](https://www.tensorflow.org/)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-lightgrey?logo=googlelens&logoColor=white&labelColor=0d1117&style=flat)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ECC71?labelColor=0d1117&style=flat)](https://choosealicense.com/licenses/mit/)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/sabneet-bains/Computer-Vision)

**Seeing patterns through pixels.**  
<sup>*A modular suite of vision systems fusing classical OpenCV pipelines with modern deep learning — for robust, real-time visual understanding.*</sup>

<img src="https://github.com/sabneet-bains/Computer-Vision/blob/main/bounded2.gif" alt="Computer Vision Example" width="800">

</div>

> [!NOTE]
> <sup>Part of the <b>Foundational & Academic</b> collection — educational tools designed with engineering rigor.</sup>


## 🧭 Table of Contents
- [Overview](#-overview)
- [Project Highlights](#-project-highlights)
- [Repository Structure](#-repository-structure)
- [Architecture](#-architecture)
- [Screenshots](#-screenshots)
- [Requirements](#-requirements)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Future Work](#-future-work)
- [Author](#-author)
- [License](#-license)


## 🧠 Overview
This repository explores advanced **computer vision** algorithms that integrate **classical image processing** with **deep learning** models.  
Each system is built for **modularity**, **reproducibility**, and **real-time performance**, making it suitable for both research and production deployment.

### 🪙 **Coin Detection & Classification**
Real-time detection and valuation of coins in live video.  
Combines adaptive preprocessing (**CLAHE**, **Gaussian blur**), **background subtraction**, **Hough circle transforms**, and **contour analysis** with a **CNN classifier** — improving accuracy by ~10% over heuristic baselines.  
Includes a **classical fallback mode** when no CNN is available.

### 🧩 **Other Image Processing Modules**
Color space transforms, morphological filtering, and edge detection experiments — forming a versatile toolkit for prototyping and experimentation.

<div align="right">

[![Back to Top](https://img.shields.io/badge/-⫛_TO_TOP-0d1117?style=flat)](#readme-top)

</div>


## 🌌 Project Highlights

| ⚙️ Module | 🧮 Description | 🧩 Framework |
|:----------|:---------------|:-------------|
| 🪙 **Coin Detection** | Real-time detection, classification, and valuation | OpenCV, TensorFlow |
| 🎨 **Image Filtering** | CLAHE, color masking, morphology, edge analysis | OpenCV, NumPy |
| 🎥 **Object Tracking (Planned)** | Multi-object tracking and motion analysis | Deep SORT, Kalman Filter |

> [!TIP]
> Projects are designed for **incremental experimentation** — start with classical CV, then extend to CNNs or hybrid pipelines.

<div align="right">

[![Back to Top](https://img.shields.io/badge/-⫛_TO_TOP-0d1117?style=flat)](#readme-top)

</div>


## 📂 Repository Structure
````text
Computer-Vision/
│
├── CoinDetection/
│   ├── coin_detect.py
│   ├── cnn_classifier.py
│   ├── utils.py
│   └── model/
│       └── trained_cnn.h5
│
├── Filters/
│   ├── color_space.py
│   ├── morphology.py
│   ├── edge_detection.py
│
├── Data/
│   ├── sample_videos/
│   └── sample_images/
│
└── README.md
````
> [!TIP]
> Directory layout mirrors **functionality × learning depth** — classical modules coexist with deep-learning counterparts for side-by-side benchmarking.

<div align="right">

[![Back to Top](https://img.shields.io/badge/-⫛_TO_TOP-0d1117?style=flat)](#readme-top)

</div>


## ⚙️ Requirements
````text
Python >= 3.9
opencv-python >= 4.8
tensorflow >= 2.10
numpy >= 1.24
````
> [!IMPORTANT]
> GPU acceleration (CUDA/cuDNN) is **optional but recommended** for CNN-based classifiers.

<div align="right">

[![Back to Top](https://img.shields.io/badge/-⫛_TO_TOP-0d1117?style=flat)](#readme-top)

</div>


## 🧪 Testing
````bash
# Run basic functional test
python CoinDetection/coin_detect.py --test

# Planned integration
pytest tests/
````

> [!NOTE]
> Automated tests (e.g., **pytest**) are under development. Contributions adding regression or performance tests are highly encouraged.

<div align="right">

[![Back to Top](https://img.shields.io/badge/-⫛_TO_TOP-0d1117?style=flat)](#readme-top)

</div>


## 🤝 Contributing
**Contributions welcome!**  
To maintain modular clarity and research reproducibility:

1. **Discuss major changes** — open an issue first.  
2. **Follow coding standards** — use docstrings, type hints, and consistent naming.  
3. **Add reproducibility evidence** — logs, screenshots, or performance metrics.  
4. **Open a pull request** with concise change notes.

> [!TIP]
> High-impact contributions include **object tracking modules**, **noise robustness experiments**, and **cross-framework benchmarks**.

<div align="right">

[![Back to Top](https://img.shields.io/badge/-⫛_TO_TOP-0d1117?style=flat)](#readme-top)

</div>


## 🚀 Future Work
- **Object Tracking:** Integrate Kalman Filters and Deep SORT for temporal consistency.  
- **New Deep Learning Models:** Broaden CNN architectures for improved coin classification.  
- **Real-Time Dashboard:** Develop live visual monitoring tools.  
- **Automated CI/CD:** Introduce test pipelines with continuous benchmarking.  
- **Expanded Docs:** Add tutorials, architecture diagrams, and Jupyter demos.

<div align="right">

[![Back to Top](https://img.shields.io/badge/-⫛_TO_TOP-0d1117?style=flat)](#readme-top)

</div>


<div align="center">

##
### 👤 Author  
**Sabneet Bains**  
*Quantum × AI × Scientific Computing*  
[LinkedIn](https://www.linkedin.com/in/sabneet-bains/) • [GitHub](https://github.com/sabneet-bains)

##
### 📄 License  
Licensed under the [MIT License](https://choosealicense.com/licenses/mit/)

<sub>“Vision reminds us — recognition isn’t perception; it’s understanding through structure.”</sub>

</div>
