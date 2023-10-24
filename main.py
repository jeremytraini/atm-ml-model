import pandas as pd
import sys
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


if len(sys.argv) != 3:
    print('Usage: python main.py <train.tsv> <test.tsv>')
    sys.exit(1)

### Part I: Regression ###

print('Running regression model...')

# Load the train and test data
train_df = pd.read_csv(sys.argv[1], delimiter='\t', header=0)
test_df = pd.read_csv(sys.argv[2], delimiter='\t', header=0)

# Remove duplicates in training data
train_df = train_df.drop_duplicates(keep='first')
train_df = train_df.dropna()

def process_pre_dummies(df):
    # Dropping irrelevant columns and columns with high correlation to others
    df.drop(columns=['ATM_Location_TYPE', 'ATM_Placement', 'ATM_TYPE', 'ATM_looks'], inplace=True)

def process_post_dummies(df):
    # Dropping ATM_Zone_RL because it has a high correlation with ATM_Zone_RM
    df.drop(columns=['ATM_Zone_RL'], inplace=True)

X_test = test_df.drop(columns=['revenue'])
y_test = test_df['revenue']

X_train = train_df.drop(columns=['revenue'])
y_train = train_df['revenue']

process_pre_dummies(X_train)
process_pre_dummies(X_test)

X_train = pd.get_dummies(data=X_train, drop_first=True)
X_test = pd.get_dummies(data=X_test, drop_first=True)

process_post_dummies(X_train)
process_post_dummies(X_test)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

corr, _ = pearsonr(y_test, y_pred)

# print('Accuracy: %.4f' % model.score(X_test, y_test))
print('Pearsons correlation: %.4f' % corr)
print()

output = pd.DataFrame({'predicted_revenue': y_pred})
output.to_csv('PART1.output.csv', index=False)


### Part II: Classification ###

print('Running classification model...')

def encode_features(X_train, X_test):
    categorical_features = ['ATM_Zone', 'ATM_Placement', 'ATM_TYPE', 'ATM_Location_TYPE', 'ATM_looks', 'ATM_Attached_to', 'Day_Type']
    encoder = LabelEncoder()
    for feature in categorical_features:
        # Fitting only with training data so future data is encoded properly
        encoder.fit(X_train[feature])
        X_train[feature] = encoder.transform(X_train[feature])
        X_test[feature] = encoder.transform(X_test[feature])


X_train = train_df.copy(True)
X_test = test_df.copy(True)

encode_features(X_train, X_test)

y_train = X_train['rating']
X_train = X_train.drop(columns=['rating'])

y_test = X_test['rating']
X_test = X_test.drop(columns=['rating'])

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=20)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

print('Accuracy: %.4f' % rf.score(X_test, y_test))
# print('F1 score (micro): %.4f' % f1_score(y_test, rf_predictions, average='micro'))

output = pd.DataFrame({'predicted_rating': rf_predictions})
output.to_csv('PART2.output.csv', index=False)
