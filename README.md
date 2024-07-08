# Healthcare Reviews Sentiment Analysis

This project focuses on sentiment analysis of healthcare reviews using machine learning techniques. The goal is to predict the sentiment (rating) based on the textual content of the reviews.

## Project Overview

The project involves the following steps:

1. **Data Loading**: The dataset (`healthcare_reviews.csv`) containing review texts and ratings is loaded using pandas.

2. **Data Cleaning**: The `preprocess_text` function is applied to clean the review texts. This includes converting text to lowercase, removing numbers, punctuation, and stop words using NLTK.

3. **Data Splitting**: The dataset is split into training and test sets using `train_test_split` from scikit-learn.

4. **Feature Engineering**: Text data is vectorized using TF-IDF (`TfidfVectorizer`) to convert text into numerical features.

5. **Model Training**: A Naive Bayes classifier (`MultinomialNB`) is trained on the TF-IDF transformed data.

6. **Model Evaluation**: Various evaluation metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score are computed to assess the performance of the model.

7. **Visualization**: 
   - A confusion matrix is plotted to visualize the model's performance in predicting ratings.
   - Sentiment distribution across different ratings is visualized using a bar chart.

## Files

- `healthcare_reviews.csv`: Dataset containing review texts and ratings.
- `sentiment_analysis.py`: Python script containing the code for data preprocessing, model training, evaluation, and visualization.

## Requirements

- Python 3.x
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Clone the repository:

git clone https://github.com/your-username/healthcare-reviews-sentiment-analysis.git

  
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



