# ATM Revenue Prediction

---

This project uses regression and classification techniques to predict revenue and rating of ATMs based on various features.

---

## Requirements
- Python 3.x
- Pandas
- Scipy
- Sklearn

---

## Usage

To run the project, use the following command:
```bash
python main.py <train.tsv> <test.tsv>
```
Replace `<train.tsv>` and `<test.tsv>` with the paths to your training and test data files, respectively.

---

## Part I: Regression
The regression model predicts revenue of ATMs based on various features. The steps involved are:

1. Loading the train and test data
2. Removing duplicates and missing values from the training data
3. Processing pre-dummies (removing irrelevant columns and columns with high correlation to others)
4. Converting categorical variables into dummy variables using one-hot encoding
5. Training a Random Forest Regressor model on the processed data
6. Predicting revenue

---

## Part II: Classification
The classification model predicts rating of ATMs based on various features. The steps involved are:

1. Loading the train and test data
2. Removing duplicates and missing values from the training data
3. Processing pre-dummies (removing irrelevant columns and columns with high correlation to others)
4. Converting categorical variables into dummy variables using one-hot encoding
5. Training a Random Forest Classifier model on the processed data
6. Predicting rating

---

## Results
The a report on the results of my findings can be found in the `report.pdf` file.
