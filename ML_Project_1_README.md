## README.md
## Healthcare Review Sentiment Analysis

## Project Overview
This project aims to analyze healthcare reviews to determine the sentiment (Positive, Neutral, or Negative) associated with each review. It involves various stages of data preprocessing, natural language processing (NLP), sentiment classification, and visualization of results. Several machine learning models are applied to classify the sentiment of reviews, including Naive Bayes, Logistic Regression, Random Forest, Support Vector Classifier (SVC), and Gradient Boosting Classifier.

## Table of Contents
1.	Installation
2.	Dataset
3.	Data Preprocessing
4.	Feature Engineering
5.	Model Training
6.	Evaluation
7.	Hyperparameter Tuning
8.	Visualization
9.	Conclusion

## Installation
To run this project, you need to install the following libraries:
bash
Copy code
pip install pandas numpy nltk sklearn matplotlib seaborn wordcloud textblob plotly imblearn

## Dataset
The dataset used in this project is a collection of healthcare reviews. The key columns are:
•	Review_text: The text of the review.
•	Rating: Rating given by the reviewer.
•	cleaned_text: Preprocessed text data.
•	Sentiment: Categorized sentiment based on the rating (Positive, Neutral, Negative).
Ensure the dataset is in CSV format and can be loaded using:
python
Copy code
df = pd.read_csv('path_to_your_healthcare_reviews.csv')

## Data Preprocessing
The preprocessing steps involve:
1.	Cleaning the text by removing special characters and stopwords.
2.	Lemmatization of words for better analysis.
3.	Sentiment polarity scoring using TextBlob.
4.	Sentiment labeling based on rating and polarity.

## Feature Engineering
•	Word Cloud: Generated to visualize the most frequent words in reviews classified as Positive, Neutral, and Negative.
•	POS Tagging: Applied to understand the structure of the sentences.

## Model Training
Five machine learning models were trained and evaluated for sentiment classification:
1.	Naive Bayes Classifier: For baseline performance.
2.	Logistic Regression: A linear model for classification.
3.	Support Vector Classifier (SVC): A powerful classifier using the SVM algorithm.
4.	Random Forest Classifier: An ensemble method using multiple decision trees.
5.	Gradient Boosting Classifier: Another ensemble method focusing on improving weak learners.

## Evaluation
The models were evaluated based on several metrics:
•	Accuracy: The ratio of correctly predicted sentiments to total predictions.
•	Precision, Recall, F1 Score: Evaluates the model's performance on positive/negative sentiment.
•	Confusion Matrix: Visualizes the classification performance.
•	ROC-AUC: Measures the model's ability to distinguish between classes.

## Hyperparameter Tuning
•	Random Forest: Hyperparameter tuning was performed using GridSearchCV to find the best parameters for improving model accuracy.

## Visualization
•	Word Clouds: For positive, negative, and neutral sentiments.
•	Bar Plots: Showing sentiment distribution.
•	Pie Charts: To depict the proportion of each sentiment.
•	Confusion Matrix: Displaying the performance of the Logistic Regression model.

## Conclusion
This project demonstrates how text data from healthcare reviews can be used to classify sentiment effectively using various NLP techniques and machine learning models. The insights derived from the visualizations help understand the key aspects driving sentiment in reviews.

## License
This project is open-source and free to use.

