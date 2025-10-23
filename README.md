# 🚀 AI-Powered Intrusion Detection System (IDS)

A real-time, intelligent Intrusion Detection System (IDS) that leverages **Machine Learning** and **AI-based traffic analysis** to detect and classify cyber threats from live network flows.  
This project is part of the **Graduation Project** focused on integrating **network security, AI, and system automation** for next-generation intrusion prevention.

---

## 🧠 Overview

The **AI-Powered IDS** analyzes network traffic using flow-based features extracted via **CICFlowMeter** and classifies packets into **benign or malicious categories** using trained machine learning models.  
It supports both **offline dataset training** (CICIDS 2018) and **real-time inference** on live traffic captures.

---

## ⚙️ Key Features

- 🌐 **Real-Time Traffic Monitoring** – captures and analyzes live network packets  
- 🤖 **Machine Learning–Driven Detection** – detects DDoS, PortScan, Brute Force, Botnet, and more  
- 📊 **Automated Data Pipeline** – preprocesses and extracts network flow features using CICFlowMeter  
- ⚡ **FastAPI Backend** – deploys the trained model for real-time classification  
- 🧩 **Modular Project Structure** – clean separation of data, models, and services  
- 🧠 **Support for Multiple Algorithms** – Random Forest, XGBoost, and Deep Neural Networks  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | Python 3.10+ |
| Framework | FastAPI |
| ML Libraries | Scikit-learn, XGBoost, TensorFlow |
| Data Processing | Pandas, NumPy |
| Network Flow Extraction | CICFlowMeter 4.0 |
| Deployment | Uvicorn, Docker (optional) |
