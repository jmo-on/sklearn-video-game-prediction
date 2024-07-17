import pandas as pd
import catboost as cat
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
vgscore_data = pd.read_csv("vgscore.csv")

# Clean data frame
vgscore_data = vgscore_data.drop(columns=["players"])

# Select independent and dependent variable
X = vgscore_data.drop(columns=['score'])
y = vgscore_data['score']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Set cat features
cf = ['name', 'platform', 'r-date', 'user score', 'developer', 'genre', 'critics', 'users']

# Instantiate model
model = cat.CatBoostRegressor(random_state=100,cat_features=cf,verbose=0)

# Fit the model
model.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))