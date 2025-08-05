
# 🧠 Breast Cancer Prediction Using Machine Learning

This project uses a breast cancer dataset to build a machine learning pipeline for classification of tumors as malignant or benign. It includes preprocessing (standardization, transformation), feature selection, data visualization, and classification using a Random Forest Classifier.

---

## 📁 Dataset

- Dataset used: `Breast_cancer_dataset.csv`
- Key target variable: `diagnosis` (M = malignant, B = benign)
- Number of features after preprocessing: 11 transformed and scaled features

---

## 🔬 Data Preprocessing

1. **Initial Checks**:
   - Checked for null values, data types, and statistical summaries.
   - Converted categorical target (`diagnosis`) into numerical codes.

2. **Feature Selection**:
   - Removed low-variance features using `VarianceThreshold(threshold=0.1)`.

3. **Standardization**:
   - Standardized features using `StandardScaler()`.

4. **Transformation**:
   - Applied `PowerTransformer(method='yeo-johnson')` for normalization.

5. **Visualization**:
   - KDE histograms for each selected feature to analyze distribution.
   - Correlation heatmaps for feature dependency and multicollinearity.

---

## 🤖 Model

- **Model Used**: `RandomForestClassifier`
- **Evaluation Strategy**: 5-Fold Cross-Validation using `cross_val_predict`
- **Pipeline**: Created using `sklearn.pipeline.Pipeline`

### 📊 Evaluation Metrics

Evaluated using `classification_report`:
- Accuracy
- Precision
- Recall
- F1-score

---

## 🛠 How to Run

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn

# Run your Python script or Jupyter Notebook with the above workflow
```

---

## 📌 Results

The final model (Random Forest with default parameters) achieved high accuracy and F1-score for both classes. This confirms that the feature engineering and preprocessing steps significantly contributed to improving prediction quality.

---

## 📚 Future Work

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Compare with other classifiers (e.g., SVM, Gradient Boosting)
- Save and deploy model using `joblib` or `pickle`

---

## 📧 Contact

Feel free to reach out for collaboration or suggestions!
