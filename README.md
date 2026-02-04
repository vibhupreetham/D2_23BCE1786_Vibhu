# 🔐 Real-Time Network Threat Detection System Using Machine Learning

A full-stack **Intrusion Detection System (IDS)** that monitors network traffic in real time, detects malicious activities using machine learning, and provides actionable alerts through a dashboard. The system is designed to be scalable, cloud-ready, and suitable for real-world deployment.

---

## 📌 Problem Statement

Traditional security systems often rely on static rules and signature-based detection, making them ineffective against modern and zero-day attacks.  
This project aims to build a **real-time, intelligent IDS** that can automatically learn network behavior patterns and detect anomalies or intrusions with high accuracy.

---

## 🎯 Objectives

- Capture and analyze real-time network traffic
- Detect malicious activity using ML models
- Classify network traffic as **Normal** or **Attack**
- Provide real-time alerts and visual analytics
- Enable cloud-based deployment (AWS integration)

---

## 🛠️ Tech Stack

### 🔹 Backend
- Python
- Flask / FastAPI
- Scapy / PyShark (packet capture)
- REST APIs

### 🔹 Machine Learning
- Scikit-learn
- TensorFlow / PyTorch (optional)
- Algorithms:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Isolation Forest (Anomaly Detection)

### 🔹 Frontend
- React.js / HTML, CSS, JavaScript
- Chart.js / Recharts (visualizations)

### 🔹 Database
- PostgreSQL / MongoDB

### 🔹 Cloud & DevOps
- AWS EC2
- AWS S3
- AWS RDS
- Docker (optional)

---

## 🧠 Machine Learning Workflow

1. **Dataset Collection**
   - NSL-KDD / CICIDS 2017 / Custom captured traffic
2. **Data Preprocessing**
   - Feature selection
   - Normalization
   - Label encoding
3. **Model Training**
   - Supervised & unsupervised learning
4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
5. **Real-Time Prediction**
   - Live packet classification
   - Alert generation

---

## 🧪 Features

- ✅ Real-time packet capture
- ✅ Attack detection (DoS, Probe, R2L, U2R)
- ✅ ML-based classification
- ✅ Dashboard with traffic statistics
- ✅ Alert & logging system
- ✅ Cloud deployment support

---

## 📊 System Architecture

```text
Network Traffic
       ↓
Packet Capture Module
       ↓
Feature Extraction
       ↓
ML Detection Engine
       ↓
Alert System + Dashboard
       ↓
Cloud Storage (AWS)
