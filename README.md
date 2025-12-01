# CalliShape Analyzer | 书法体势分析器 🖌️

**CalliShape Analyzer** is a computer vision tool designed for Chinese calligraphy analysis. It automatically detects characters, draws precise bounding boxes, and classifies each character's shape (Long, Square, or Flat) based on its aspect ratio.

**CalliShape Analyzer** 是一个专为中国书法分析设计的计算机视觉工具。它能够自动检测每一个汉字，绘制精准的边缘框，并根据“长、方、扁”的体势规则对汉字进行分类和着色标注。

![Demo]([https://via.placeholder.com/800x400?text=Place+Your+Result+Image+Here](https://github.com/blackhost-origin/CalliShape_Analyzer/blob/main/precise_calligraphy_boxes.jpg)
*(Please replace this link with your actual result image / 请替换为你的实际运行结果图)*

## ✨ Key Features (核心功能)

* **Intelligent Character Detection (智能单字识别)**:
    * Uses **Morphological Closing (闭运算)** to correctly group disjointed strokes (e.g., left-right structures like "明" or "川") into a single character box.
    * 利用**形态学闭运算**将分离的笔画（如左右结构的字）智能粘合，避免将一个字识别为多个部分。
* **Anti-Nesting Logic (去嵌套保护)**:
    * Automatically removes inner bounding boxes (e.g., the space inside "口" or "周") to ensure only the outer boundary is captured.
    * 自动剔除嵌套在内部的小框，确保每个汉字只保留一个最外层的最大矩形。
* **Shape Classification & Visualization (体势分类与可视化)**:
    * Classifies characters based on Aspect Ratio (Height/Width):
        * 🔴 **Long (长)**: Ratio > 1.2 (Red Box)
        * 🟢 **Square (方)**: 0.8 ≤ Ratio ≤ 1.2 (Green Box)
        * 🔵 **Flat (扁)**: Ratio < 0.8 (Blue Box)
* **Robust Pre-processing (强鲁棒性预处理)**:
    * Uses OTSU binarization to handle various paper textures and ink densities.
    * 采用 OTSU 自适应二值化，适应不同纸张背景和墨色浓淡。

## 🛠️ Dependencies (依赖库)

Ensure you have Python 3.x installed. Install the required libraries using pip:

```bash
pip install opencv-python numpy matplotlib
