import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression

# Load the train and test data
train_df = pd.read_csv('train.tsv', delimiter='\t', header=0)
test_df = pd.read_csv('test.tsv', delimiter='\t', header=0)


print(train_df.head())

corr_matrix = train_df.corr()
print(corr_matrix["revenue"].sort_values(ascending=False))


features = ["No_of_Other_ATMs_in_1_KM_radius", "Estimated_Number_of_Houses_in_1_KM_Radius", "rating"]

X = train_df[features]
y = train_df["revenue"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print("MSE:", mse)


# drop Number_of_Shops_Around_ATM
train_df = train_df.drop(columns=['Number_of_Shops_Around_ATM'])


# select the features from the test set
X_test = test_df[features]

# predict the revenue on the test set
y_test_pred = model.predict(X_test)

# print the predicted revenue
print(y_test_pred)

mse = mean_squared_error(X_val, y_test_pred)
print("MSE:", mse)


