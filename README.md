# Artificial Neural Networks (ANN) - Lab Task #02
## Binary Classification using Logistic Regression & Web-Based Predictive Interfaces

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg?logo=flask)](https://palletsprojects.com/p/flask/)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive%20UI-orange.svg?logo=gradio)](https://gradio.app/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Institution](https://img.shields.io/badge/SZABIST-Islamabad-006699.svg)](https://szabist-isb.edu.pk/)

---

### 🎓 Academic Information
- **Student Name:** Aleeya Fatima
- **Registration Number:** 24108108
- **Program:** BS Artificial Intelligence (BS AI - 5A)
- **Department:** Department of Robotics & Artificial Intelligence
- **Course:** Artificial Neural Networks (Lab)
- **Submitted To:** Mr. Hassaan Mujtaba
- **Institution:** Shaheed Zulfikar Ali Bhutto Institute of Science & Technology (SZABIST), Islamabad

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Task 1: Flood Prediction System](#task-1-flood-prediction-system)
   - [Problem Definition](#problem-definition)
   - [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
   - [Model Architecture & Training](#model-architecture--training)
   - [Evaluation Metrics & Results](#evaluation-metrics--results)
   - [Interactive Interfaces (Flask & Gradio)](#interactive-interfaces-flask--gradio)
4. [Task 2: Climate Change Economic Impact Predictor](#task-2-climate-change-economic-impact-predictor)
   - [Problem Definition](#problem-definition-1)
   - [Target Formulation](#target-formulation)
   - [Model Training & Analysis of Data Limitations](#model-training--analysis-of-data-limitations)
   - [Evaluation Metrics & Results](#evaluation-metrics--results-1)
5. [Unified Predictive Portal](#unified-predictive-portal)
6. [Installation & Setup](#installation--setup)
7. [Running the Projects](#running-the-projects)

---

## 🔍 Overview
This repository contains the complete implementation for **ANN Lab Task #02**, focusing on supervised binary classification using **Logistic Regression**, feature standardization using **StandardScaler**, robust train-test validation (75/25), and deployment of interactive user interfaces using **Flask** and **Gradio**.

Two real-world classification challenges are solved:
1. **Flood Occurrence Classification**: Predicting whether a flood will occur based on meteorological rainfall and topographical indicators.
2. **Climate Economic Impact Classification**: Predicting whether the economic consequences of global climate events will be High or Low relative to median impact benchmarks.

---

## 📁 Repository Structure

```text
├── ANN_Lab_Task_2_24108108.ipynb      # Complete Jupyter Notebook with code, outputs, and UI
├── Ann lab task2 24108108.pdf         # Official Lab Report Submission Document
├── app.py                             # Unified Flask Web Portal combining both models
├── task1_flood.py                     # Standalone CLI training and testing for Flood Prediction
├── task2_climate.py                   # Standalone CLI training and testing for Climate Impact
├── flask_flood_app.py                 # Dedicated Flask web app for Flood Prediction (Port 5004)
├── flask_climate_app.py               # Dedicated Flask web app for Climate Impact (Port 5005)
├── flood_dataset_classification.csv   # Dataset for Task 1 (6,240 rows)
├── realistic_climate_change_impacts.csv# Dataset for Task 2 (10,000 rows)
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git ignore patterns
└── README.md                          # Comprehensive documentation
```

---

## 🌊 Task 1: Flood Prediction System

### Problem Definition
Predict whether a geographic region will experience a flood (`1 = Yes`, `0 = No`) from terrain and rainfall metrics.

- **Dataset:** `flood_dataset_classification.csv` (6,240 samples)
- **Target Variable:** `occured` (Binary: 0 or 1)
- **Input Features:**
  - `Rainfall`: Precipitation volume (mm)
  - `Elevation`: Height above sea level (meters)
  - `Slope`: Topographical incline/gradient
  - `distance`: Proximity to the nearest river (meters)

### Feature Engineering & Preprocessing
Because terrain features vary widely in magnitude (e.g., river distance ranges in thousands of meters while slopes are single digits), **StandardScaler** was used:
$$z = \frac{x - \mu}{\sigma}$$
This standardizes each feature to have zero mean and unit variance, preventing high-magnitude features from dominating gradient updates.

### Model Architecture & Training
- **Algorithm:** Logistic Regression (Scikit-Learn)
- **Data Partition:** 75% Training (4,680 samples), 25% Testing (1,560 samples) with `random_state=7`
- **Activation:** Sigmoid function $P(y=1) = \frac{1}{1 + e^{-z}}$

### Evaluation Metrics & Results
- **Overall Accuracy:** **70.71%** (~71%)
- **Confusion Matrix:**
  ```text
  [[ 114  412]
   [  45  989]]
  ```
- **Classification Report:**
  | Class | Precision | Recall | F1-Score | Support |
  | :--- | :---: | :---: | :---: | :---: |
  | **No Flood (0)** | 0.72 | 0.22 | 0.33 | 526 |
  | **Flood (1)** | 0.71 | 0.96 | 0.81 | 1,034 |
  | **Accuracy** | | | **0.71** | 1,560 |
  | **Weighted Avg** | 0.71 | 0.71 | 0.65 | 1,560 |

---

## 🌍 Task 2: Climate Change Economic Impact Predictor

### Problem Definition
Predict whether the economic toll of a global climate anomaly will be **High (1)** or **Low (0)**.

- **Dataset:** `realistic_climate_change_impacts.csv` (10,000 samples)
- **Input Features:**
  - `TemperatureAnomaly_C`: Deviation in surface temperature (°C)
  - `CO2Level_ppm`: Atmospheric carbon dioxide concentration (ppm)
  - `PopulationAffected`: Number of vulnerable citizens impacted

### Target Formulation
Since the dataset provides continuous dollar losses (`EconomicImpact_USD`), a binary classification boundary was engineered using the median economic impact:
$$\text{HighImpact} = \begin{cases} 1, & \text{if } \text{EconomicImpact\_USD} > \text{Median} \\ 0, & \text{otherwise} \end{cases}$$

### Model Training & Analysis of Data Limitations
- **Algorithm:** Logistic Regression
- **Data Partition:** 75% Training (7,500 samples), 25% Testing (2,500 samples) with `random_state=7`
- **Feature Scaling:** `StandardScaler`
- **Accuracy:** **50.52%** (~51%)
- **Confusion Matrix:**
  ```text
  [[703 532]
   [705 560]]
  ```

> **Analytical Note on Model Performance:**  
> The resulting ~50.5% accuracy is directly attributable to the synthetic generation distribution of this dataset, where the environmental predictor columns exhibit near-zero correlation with the resulting economic impact. This demonstrates an essential machine learning principle: model capacity cannot extract predictive signal where statistical dependency does not exist in the raw data.

---

## 🚀 Unified Predictive Portal (`app.py`)
A single unified web portal is provided in `app.py`, featuring a modern dark-mode glassmorphic user interface built with HTML5/CSS3 and Flask. It allows interactive exploration of both models simultaneously with real-time risk classification and probability output.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AleeyaFatima/ANN_LAB2.git
   cd ANN_LAB2
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Projects

### Option 1: Jupyter Notebook
Launch Jupyter Notebook to view explanations, code, and interactive Gradio blocks:
```bash
jupyter notebook ANN_Lab_Task_2_24108108.ipynb
```

### Option 2: Standalone Command-Line Scripts
Train and evaluate models directly in terminal:
```bash
# Task 1: Flood Prediction
python3 task1_flood.py

# Task 2: Climate Impact Prediction
python3 task2_climate.py
```

### Option 3: Dedicated Flask Web Applications
Run individual web interfaces:
```bash
# Run Flood Web App on http://127.0.0.1:5004
python3 flask_flood_app.py

# Run Climate Web App on http://127.0.0.1:5005
python3 flask_climate_app.py
```

### Option 4: Unified Web Portal
Launch the consolidated dashboard on port 5000:
```bash
python3 app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📊 Summary Comparison

| Metric / Parameter | Task 1: Flood Prediction | Task 2: Climate Economic Impact |
| :--- | :--- | :--- |
| **Model** | Logistic Regression | Logistic Regression |
| **Problem Type** | Binary Classification (`0` or `1`) | Binary Classification (`0` or `1`) |
| **Dataset Size** | 6,240 records | 10,000 records |
| **Train/Test Split** | 75% / 25% | 75% / 25% |
| **Preprocessing** | StandardScaler | StandardScaler |
| **Test Accuracy** | **70.71%** (~71%) | **50.52%** (~51%) |
| **Primary Class F1** | 0.81 (Flood class) | 0.48 (High impact class) |
| **Web Interface** | Gradio + Flask (Port 5004) | Gradio + Flask (Port 5005) |

---
**SZABIST Islamabad — Department of Robotics & Artificial Intelligence**
