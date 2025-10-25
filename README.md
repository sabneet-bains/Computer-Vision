# 👁️ Computer Vision Repository  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-lightgrey?logo=googlelens&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

<br>

**A curated collection of computer vision and image processing systems developed in Python, combining classical OpenCV techniques and deep learning for robust, real-time performance under challenging conditions.**

<img src="https://github.com/sabneet95/Computer-Vision/blob/main/bounded2.gif" alt="Computer Vision Example" width="800">


## 🧭 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Highlighted Projects](#highlighted-projects)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Testing](#testing)
- [Contributing](#contributing)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)


## 🧩 Overview

This repository explores and implements advanced computer vision algorithms that integrate **classical image processing** with **deep learning**.  
Projects emphasize reproducibility, real-time performance, and modular design for research and production environments.

### **Coin Detection and Classification**
A live detection system that processes video to identify, classify, and assign values to coins in real time.  
It integrates adaptive preprocessing (CLAHE, Gaussian blur), background subtraction, Hough circle transforms, and contour analysis with a custom CNN to achieve robust detection and ~10% accuracy improvements over heuristic methods.  
A classical fallback mode is included when no CNN is provided.

### **Other Image Processing Modules**
Additional modules include experiments in color space transformations, morphological filtering, and edge detection — forming a versatile toolkit for both research and applied development.


## 🧱 Architecture

The repository is designed with modularity and extensibility in mind:

- **Classical CV Techniques →** Implements proven methods such as Hough transforms, background subtraction, and contour analysis for feature extraction.  
- **Deep Learning Integration →** Optional CNNs (TensorFlow/Keras) enhance classification accuracy and adaptability.  
- **Robust Preprocessing →** Adaptive algorithms like CLAHE ensure consistent performance under fluctuating lighting.  
- **Separation of Concerns →** Each module (e.g., coin detection, color transformations) is independently testable and integrable.


## 🧠 Highlighted Projects

| Project | Description | Key Technologies |
|----------|--------------|------------------|
| **Coin Detection** | Real-time detection, classification, and valuation of coins | OpenCV, TensorFlow, CNN |
| **Image Filtering** | CLAHE, color masking, morphology, edge analysis | OpenCV, NumPy |
| **Object Tracking (Planned)** | Multi-object tracking and motion analysis | Deep SORT, Kalman Filter |


## 🖼️ Screenshots

<img src="https://github.com/sabneet95/Computer-Vision/blob/main/curves.jpg" alt="Screenshot 1" width="800">
<img src="https://github.com/sabneet95/Computer-Vision/blob/main/sith.jpg" alt="Screenshot 2" width="800">
<img src="https://github.com/sabneet95/Computer-Vision/blob/main/bounded2.gif" alt="Screenshot 3" width="800">


## ⚙️ Requirements

- **Python 3.9.1 or later (64-bit)**  
  [Download Python](https://www.python.org/downloads/)  
- **OpenCV 4.x**  
  [OpenCV Documentation](https://docs.opencv.org/4.x/)  
- **TensorFlow/Keras**  
  For running CNN-based classifiers.  
- *(Optional)* **CUDA-enabled GPU**  
  For accelerated processing.


## 🧪 Testing

<details>
<summary>Testing Status</summary>

Automated tests are not yet integrated.  
Future updates may introduce **pytest**-based regression and performance testing.  
Contributions to improve test coverage are welcome.
</details>


## 🤝 Contributing

Contributions are welcome!  

1. **Discuss Major Changes** — open an issue before implementing large features.  
2. **Follow Coding Standards** — document functions clearly and maintain modular consistency.  
3. **Submit Pull Requests** — include concise change descriptions and any new tests or examples.

> 💡 Contributors working on **OpenCV extensions**, **deep learning integration**, or **real-time optimization** are especially encouraged to participate.


## 🚀 Future Work

Planned enhancements include:

- **Enhanced Object Tracking:** Integration of Kalman Filters and Deep SORT for temporal consistency.  
- **Additional Deep Learning Models:** Broaden support for coin and object classification under variable conditions.  
- **Real-Time Dashboard:** Develop a live monitoring dashboard for data visualization.  
- **Automated Testing:** Integrate continuous testing and CI pipelines.  
- **Expanded Documentation:** Provide detailed tutorials, architecture diagrams, and example use cases.


## 🧠 Author

**Sabneet Bains** — *Quantum × AI × Scientific Computing*  
[LinkedIn](https://www.linkedin.com/in/sabneet-bains/) • [GitHub](https://github.com/sabneet-bains)


## 📄 License

This repository is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

