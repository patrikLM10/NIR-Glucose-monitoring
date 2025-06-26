# Non-Invasive Glucose Prediction Using ML Models

This repository implements a non-invasive blood glucose prediction system using near-infrared (NIR) spectroscopy data. It showcases three machine learning approaches:

* **Feed-Forward Neural Network (FFNN)**
* **LightGBM Regressor**
* **CatBoost Regressor**

All models are trained on dual-wavelength NIR features and evaluated against a reference finger-prick glucometer.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/glucose-ml.git
cd glucose-ml
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data


* Format: Each row should contain an 8-dimensional feature vector (NIR-derived features for 940 nm & 950 nm) along with the corresponding glucose level (mg/dL) as ground truth.

---

## 🧠 Models & Training

### FFNN (Feed‑Forward Neural Network)

* **Architecture:** 8‑64‑16‑1 layers, ReLU activations, linear output
* **Optimizer:** Adam, learning rate = 1e‑3
* **Loss:** Mean Squared Error (MSE) with L₂ regularization
* **Training:** 200–250 epochs, batch size 32, 20% validation split

Train with:

```bash
python scripts/train_ffnn.py --data clas.csv --output models/ffnn_model.h5
```

### LightGBM & CatBoost

* **Hyperparameter Tuning:** Optuna, 5‑fold CV, minimize RMSE
* **Final Models:** LGBMRegressor, CatBoostRegressor

Tune & train with:

```bash
python scripts/train_gbm.py --data clas.csv --output_dir models/
```

---

## 📈 Evaluation

Run predictions and compute metrics (MAE, RMSE, MAPE, R²):

```bash
python scripts/evaluate_models.py --models_dir models/ --data clas.csv
```

Sample results on test set:

| Model    | R²     | MAE (mg/dL) | RMSE (mg/dL) | MAPE (%) |
| -------- | ------ | ----------- | ------------ | -------- |
| FFNN     | 0.9986 | 2.98        | 3.63         | 2.32     |
| LightGBM | 0.9988 | 2.71        | 3.34         | 2.04     |
| CatBoost | 0.9988 | 2.69        | 3.30         | 2.02     |

All models exceed ISO 15197 clinical accuracy (± 15 mg/dL / ± 15 %).

---

## 🔍 Visualization & Analysis

* **Loss curves:** track train/validation MSE over epochs
* **True vs. Predicted:** scatter with 45° reference
* **Residual distribution:** histogram of errors
* **Bland‑Altman:** bias and limits of agreement

See `notebooks/analysis.ipynb` for full plots.

---

## 🎯 Next Steps

* Deploy the best model (e.g., CatBoost) via a microservice or mobile app.
* Integrate real-time NIR sensor data ingestion.
* Conduct prospective clinical validation with new subjects.

---

## 📄 License

Released under the MIT License. See `LICENSE` for details.

