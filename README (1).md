# Wine Quality Prediction ML Project

## Objective  
Predict the quality of red wine using physicochemical properties. This model helps wine producers and distributors estimate wine quality based on measurable chemical properties — useful for quality control, pricing, and production decisions.

## Dataset  
- *Name:* Wine Quality Dataset (Red / White)  
- *Source:* Kaggle / UCI Machine Learning Repository 3  
- *File used:* winequality-red.csv (or winequality-white.csv)  
- *Features:* fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, etc. 4  
- *Target:* quality (wine quality score)  

## Steps in Project  
1. Load the dataset safely (same folder as script).  
2. Check and clean missing values (if any).  
3. Split dataset into features (X) and target (y).  
4. Train-test split (80% training, 20% testing).  
5. Train a *Random Forest Regressor*.  
6. Evaluate model using *R² score* and *RMSE*.  
7. Visualizations:  
   - Actual vs Predicted quality distribution  
   - Top 5 most important features influencing quality  
   - Correlation heatmap of features  

## Requirements  
- Python 3.x  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  

Install dependencies:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn


