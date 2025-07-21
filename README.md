# 🫀 Heart Disease ML Project

This is a complete machine learning pipeline I built for predicting heart disease using clinical data. It’s part of my AI/ML summer course project, and I used it to apply everything I’ve learned — from preprocessing to model export and even optional deployment.

---

## 🔍 Project Goals

- Clean and prepare the dataset
- Apply PCA for dimensionality reduction
- Perform feature selection to find the most important variables
- Train and evaluate different supervised learning models
- Use unsupervised learning for clustering insights
- Tune hyperparameters for better accuracy
- Export the best model for deployment
- _(Bonus)_ Build a simple web app using Streamlit

---

## 📂 Project Structure

Heart_Disease_Project/
│── data/
│ ├── heart_disease.csv
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│ ├── 07_model_export.ipynb
│ ├── test_class_distrubtion.ipynb
│── models/
│ ├── final_model.pkl
│── ui/
│ ├── app.py (Streamlit UI)
│── deployment/
│ ├── ngrok_setup.txt
│── results/
│ ├── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore

---

## 🔍 Model Performance & Limitations

While we trained and evaluated four supervised learning models (Logistic Regression, Decision Tree, Random Forest, and SVM), the overall accuracy remained limited — with the highest accuracy around **0.54**. Upon deeper analysis, this performance is attributed to **class imbalance** and **limited data samples**, especially in higher heart disease severity levels.

---

### 📊 Class Distribution:

| Target Value | Count |
| ------------ | ----- |
| 0            | 164   |
| 1            | 55    |
| 2            | 36    |
| 3            | 35    |
| 4            | 13    |

The dataset is heavily skewed toward class `0` (no heart disease), resulting in biased learning. The models tend to perform well on class `0` but struggle significantly on minority classes (`1`, `2`, `3`, `4`), reducing the overall predictive reliability.

> ⚠️ We intentionally chose **not** to apply artificial balancing techniques like SMOTE in this project to maintain data authenticity and emphasize this limitation transparently.

Future work can explore resampling strategies or collecting more diverse data to improve the models' generalization across all target classes.

---

## 🛠️ Tools & Libraries

- Python 3.10
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- joblib
- streamlit _(optional)_

---

## 📌 Notes

This project helped me understand the complete ML process. I got to work on data cleaning, model training, evaluation, and even exporting a trained model. It was a great experience to connect everything together in one clean pipeline.
