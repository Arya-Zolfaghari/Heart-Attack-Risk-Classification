# Heart Disease Classification: Logistic Regression vs Random Forest

The objective of this project is to build and compare a Logistic Regression model and a Random Forest model in order to predict if the patient has heart disease from the provided heart.csv file. This project to includes proper scaling of features, class weighting to address class imbalance, a variety of evaluation metrics and visualizations, confusion matrices, as well as 3D visualizations of probability, and a view of feature importance.

## 📊 Dataset
- File: heart.csv
- Target Variable: output (1 = high risk / disease, 0 = low risk / no disease)
- Features: all other variables except output are used as predictive variables

## ⚙️ Pipeline
1. **Train/Test Split**
   - train_test_split(..., test_size=0.2, stratify=y, random_state=42)
2. **Logistic Regression**
   - StandardScaler on X
   - LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
     - Threshold at 0.5 on predict_proba
3. **Random Forest**
   - RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42)
4. **Evaluation**
   - Evaluation metric: ROC-AUC, Accuracy, F1, Classification report
   - ConfusionMatrixDisplay for both models
   - 3-D probability scatter:
     - LogReg: color by predict_proba over (age, chol, thalachh)
     - RF: color by predict_proba over top-3 important features
   - Feature Importances (RF)
5. **Model Selection**
   - Compare ROC-AUC of both models and print predicting only the best model.

## 📈 Outputs You’ll See
- Console metrics for both models (ROC-AUC, Accuracy, F1, per-class precision/recall/F1).
- Confusion matrix plots:
  - Rows = True labels, Columns = Predicted labels (`Low risk`, `High risk`)
- 3-D scatter of the probabilities of the LogReg & RF models.
- Bar chart of **Random Forest feature importances**.
- Final line: `>>> Best model based on ROC-AUC: ..`

## 🛠️ Requirements
```bash
pip install numpy pandas scikit-learn matplotlib
