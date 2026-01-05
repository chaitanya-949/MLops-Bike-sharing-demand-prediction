# End-to-End MLOps Pipeline with Databricks, AWS & CI/CD

## 📌 Project Overview

This project demonstrates a **production-grade end-to-end MLOps pipeline** where data is fetched from **Databricks**, versioned and processed locally, trained using a modular ML pipeline, and deployed using **Docker + GitHub Actions** on **AWS infrastructure**.

The system is designed with **scalability, reproducibility, and automation** in mind, following industry-standard MLOps practices.

---

## 🏗️ Architecture

**Data → Training → Model Registry → Deployment → Inference**

**High-level flow:**

1. Databricks SQL Warehouse → Data Ingestion
2. Local ML Pipeline (Validation, Transformation, Training)
3. Model versioning 
4. Model artifact stored in AWS S3
5. Dockerized FastAPI application
6. CI/CD via GitHub Actions
7. Deployed on AWS EC2

---

## ⚙️ Tech Stack

* **Data Source**: Databricks SQL Warehouse
* **Language**: Python
* **ML Model**: XGBoost Regressor
* **Pipeline Design**: Modular OOP-based pipeline
* **Experiment Tracking & Versioning**: DVC
* **Artifact Storage**: AWS S3
* **Containerization**: Docker
* **CI/CD**: GitHub Actions (Self-hosted runner)
* **Deployment**: AWS EC2
* **API Framework**: FastAPI

---

## 🔁 MLOps Pipeline Stages

### 1️⃣ Data Ingestion

* Connects to **Databricks SQL Warehouse** using secure credentials
* Executes SQL query to fetch data
* Stores raw data locally
* Performs **train-test split**

### 2️⃣ Data Validation

* Schema validation
* Required column checks
* Categorical & numerical feature validation
* Validation report generated

### 3️⃣ Data Transformation

* Feature engineering on date columns (month, day, weekday)
* One-hot encoding for categorical variables


### 4️⃣ Model Training

* XGBoost Regressor trained on transformed data
* Model evaluated using **R² score (~0.89)**
* Best-performing model selected

### 5️⃣ Model Evaluation

* New model compared against existing production model
* Automatically replaces model if performance improves

### 6️⃣ Model Versioning (DVC)

* Trained model (`model.pkl` ~60MB)
* Model artifacts pushed to **AWS S3 remote**
* Ensures full reproducibility across environments

### 7️⃣ Deployment

* FastAPI app loads model directly from S3
* Application containerized using Docker
* CI/CD pipeline builds & deploys image via GitHub Actions
* Deployed on AWS EC2 with public endpoint

---

## 🚀 CI/CD Workflow (GitHub Actions)

* Triggered on push to main branch
* Steps:

  * Code checkout
  * Build Docker image
  * Run container tests
  * Push image
  * Deploy to EC2 using self-hosted runner



---

## 🔐 Configuration & Secrets

* Databricks credentials managed via environment variables
* AWS credentials configured securely
* No secrets hard-coded in the repository

---

## 📊 Key Results

* R² Score: **~0.89**
* Fully automated training → deployment pipeline
* Production-ready deployment with CI/CD

---

## 🌟 Key Learnings

* Designing scalable ML pipelines
* Production-level logging & exception handling
* End-to-end automation with CI/CD
* Real-world cloud deployment challenges

---

## 📌 Future Improvements

* Add MLflow for experiment tracking
* Introduce monitoring & drift detection
* Add unit & integration tests

---

## 👤 Author

**Chaitanya Tulluri**
Data Scientist | MLOps Enthusiast | Applied ML Engineer
