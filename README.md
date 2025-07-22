# AI SaaS-Based Adaptive IDS Model Using Federated Learning for Privacy-Preserving Threat Detection

This repository contains the complete implementation of my MSc dissertation project, which proposes a SaaS-based adaptive Intrusion Detection System (IDS) using Federated Learning (FL). The aim is to detect threats while preserving user privacy across distributed environments.

## 📌 Project Overview

Traditional IDS models often require centralized data aggregation, posing privacy risks. This project introduces a privacy-preserving IDS model leveraging:

- **Federated Learning (FL)**: Enables decentralized training without sharing raw data.
- **Adaptive IDS Model**: Dynamically adjusts to new attack patterns.
- **Multi-dataset Evaluation**: Tested on KDDCup99 and UNSW-NB15 datasets.

## 🚀 Key Features

- ✅ Support for both **KDD** and **UNSW-NB15** datasets  
- ✅ **Federated Learning server-client architecture** using Flower (`flwr`)  
- ✅ **TensorFlow-based training pipelines** for extensibility  
- ✅ Support for multiple clients (`client1.py`, `client2.py`, `client3.py`)  
- ✅ Preprocessing scripts for each dataset  
- ✅ Dashboard logging and performance tracking

---

## 🧠 Technologies Used

- Python 3.x  
- TensorFlow  
- Flower (FL framework)  
- Pandas, NumPy, Sklearn  
- Jupyter Notebook (for testing and visualization)

---

## 📂 Repository Structure

```bash
.
├── client1.py               # Standard FL client
├── client1_TF.py            # TensorFlow-specific FL client
├── client1_UNSW.py          # UNSW-specific client implementation
├── client2.py               # Second client for FL
├── client3.py               # Third client for FL
├── server.py                # FL server logic
├── dashboard.py             # Visualization & monitoring
├── balance_nsl_kdd.py       # Balancing script for KDD dataset
├── nsl_kdd_preprocess.py    # Preprocessing KDD data
├── unsw_nb15_preprocess.py  # Preprocessing UNSW-NB15 data
├── KDDTrain+.txt            # Raw KDD training dataset
├── KDDTest+.txt             # Raw KDD test dataset
├── UNSW_NB15_testing-set.csv # Raw UNSW test data
├── NUSW-NB15_features.csv   # UNSW feature definitions
├── Server_FL_Aggregator.pem # PEM certificate (if using TLS)
├── README.md                # This file

📊 Results Summary
✅ Achieved high accuracy across multiple FL rounds with both datasets

✅ Demonstrated resilience to data distribution variations

✅ Showed effective privacy preservation without sacrificing performance

Figures, confusion matrices, and training accuracy graphs are included in the final dissertation document.

📘 How to Run
Install requirements
pip install -r requirements.txt

Run the FL server
python server.py

Run each client in a separate terminal
python client1.py
python client2.py
python client3.py

Monitor with dashboard
python dashboard.py


---

Let me know if you want a shorter version or want your actual email and LinkedIn links added.
www.linkedin.com/in/gull-hassan-455b2611a
