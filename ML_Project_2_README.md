# Hospital Readmissions Prediction

This project focuses on predicting hospital readmissions using a Random Forest Classifier. It involves preprocessing data, training a machine learning model, evaluating its performance, and visualizing key metrics.

## Project Overview

The project includes the following steps:

1. **Data Loading**: The dataset (`synthetic_hospital_readmissions_data.csv`) containing patient information is loaded using pandas.

2. **Handling Missing Values**: Missing values are filled with median values for numeric features and mode values for categorical features.

3. **Encoding Categorical Variables**: Categorical variables such as Gender, Admission Type, Diagnosis, A1C Result, and Readmitted are encoded using Label Encoding from scikit-learn.

4. **Feature Engineering**: Features are created or manipulated to prepare data for model training.

5. **Data Splitting**: The dataset is split into training and test sets using `train_test_split` from scikit-learn.

6. **Normalization**: Data is standardized using `StandardScaler` to ensure all features contribute equally to the model.

7. **Model Training**: A Random Forest Classifier (`RandomForestClassifier`) with 100 estimators is trained on the training data.

8. **Model Evaluation**: Various metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score are computed to evaluate the model's performance.

9. **Visualization**: 
   - Confusion matrix is plotted to visualize the true positive, false positive, true negative, and false negative predictions.
   - ROC Curve is plotted to visualize the trade-off between true positive rate and false positive rate.
   - Feature Importance is shown using a bar chart to understand which features contribute most to the model's predictions.

## Files

- `synthetic_hospital_readmissions_data.csv`: Dataset containing patient information and readmission status.
- `hospital_readmissions_prediction.py`: Python script containing the code for data preprocessing, model training, evaluation, and visualization.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Clone the repository:

2. Install dependencies:

3. Run the script:


## Results

- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1 Score**: XX%
- **ROC-AUC**: XX%

## Author

- [RAJAGANAPATHY](https://github.com/your-rajaganaa)

Feel free to contribute, report issues, or provide feedback!

