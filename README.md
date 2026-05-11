# Appliances Energy Consumption Prediction

A machine learning regression project comparing 6 algorithms to predict household appliance energy consumption from IoT sensor data (temperature, humidity, weather).

---

## Results

### Baseline Models
| Model | Train R² | Test R² | Test RMSE |
|---|---|---|---|
| Ridge Regression | 0.138 | 0.121 | 0.937 |
| Lasso Regression | 0.000 | 0.000 | 1.000 |
| Support Vector Regression | 0.236 | 0.210 | 0.889 |
| **KNN Regressor** | 0.681 | **0.486** | 0.717 |
| **Random Forest** ⭐ | **0.939** | **0.556** | **0.667** |
| MLP Regressor | 0.299 | 0.243 | 0.870 |

### After Hyperparameter Tuning (GridSearchCV, 5-fold CV)
| Model | Train R² | Test R² | Best Parameters |
|---|---|---|---|
| Ridge Regression | 0.138 | 0.122 | alpha=10.0, solver=saga |
| Lasso Regression | 0.121 | 0.113 | alpha=0.01, selection=cyclic |
| SVR | 0.173 | 0.150 | epsilon=0.1, kernel=poly |
| **KNN Regressor** ⭐ | 1.000 | **0.576** | n_neighbors=3, weights=distance |
| Random Forest | 0.943 | 0.570 | max_features=sqrt, n_estimators=200 |
| MLP Regressor | 0.509 | 0.382 | hidden_layer_sizes=(100,100), alpha=0.01 |

**Winner: KNN Regressor (Test R²: 0.576) after tuning**

---

## Dataset

- **Source:** UCI Appliances Energy Prediction Dataset
- **Size:** 19,735 rows × 29 features
- **Period:** January–May 2016 (10-minute intervals)
- **Target:** `Appliances` — energy use in Wh
- **Features:** 9 indoor temperature sensors, 9 humidity sensors, outdoor weather data (T_out, RH_out, Windspeed, etc.)

---

## Methodology

**Exploratory Analysis**
- Identified energy peaks by hour of day and day of week using heatmaps
- Applied log transformation to target variable to address right skew
- Correlation analysis revealed `rv1`/`rv2` (random noise), `T6`, `T9`, and `Visibility` had low predictive value — removed from final model

**Preprocessing**
- Outlier removal: dropped Appliances > 790 Wh
- Feature engineering: added `hour`, `weekday`, `month` from timestamp index
- Dropped `lights` column (77% zero values — insufficient signal)
- StandardScaler normalization on all features
- 75/25 train/test split (random_state=40)

**Feature Selection (final model uses 21 features)**
T1–T5, T7, T8, RH_1–RH_9, T_out, Tdewpoint, RH_out, Press_mm_hg, Windspeed

---

## What I Learned

This project showed me how much feature selection matters for regression tasks — simply dropping the four low-correlation columns (rv1, rv2, T6, T9) meaningfully improved model performance. I also learned that KNN can dramatically overfit on training data (Train R²: 1.000) while still generalizing reasonably well on test data when using distance-weighted voting, which was an interesting tradeoff. Lasso's complete failure (R²: 0.000) at default settings highlighted how sensitive L1 regularization is to hyperparameter choice — after tuning, it recovered to 0.113.

---

## Tech Stack

Python · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Google Colab

---

## Setup

```bash
git clone https://github.com/Ravneek29/Appliances-Energy-Prediction.git
cd Appliances-Energy-Prediction
pip install -r requirements.txt
# Download dataset: energydata_complete.csv from UCI ML Repository
# Open Appliances_Energy_Prediction.ipynb in Jupyter or VS Code
```
