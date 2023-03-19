import datetime

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def convert_date(date_str):
    try:
        date_obj = datetime.datetime.strptime(date_str, '%m/%d/%Y')
    except:
        date_obj = None
    if date_obj is not None:
        timestamp = int(date_obj.timestamp())
    else:
        timestamp = None
    return timestamp

# apply the function to the "ReleasedOn" column and create a new column called "ReleasedTimestamp"


# print the updated DataFrame
#print(data.head())

# Load the dataset
data = pd.read_csv('appsDataTest.csv')

data['TitleLength'] = data['Title'].apply(len)
data['DescriptionLength'] = data['Description'].apply(len)
data['ReleasedTimestamp'] = data['ReleasedOn'].apply(convert_date)
# Drop 'Id', 'Title', and 'Description' columns as they are irrelevant for our model
data = data.drop(['Id', 'Title', 'Description', 'ReleasedOn'], axis=1)

# Convert 'ReleasedOn' column to datetime
#data['ReleasedOn'] = pd.to_datetime(data['ReleasedOn'])

# Convert boolean columns to boolean data type
data['Free'] = data['Free'].astype(bool)
data['Sale'] = data['Sale'].astype(bool)
data['In_app_purchase'] = data['In_app_purchase'].astype(bool)
data['AdSupported'] = data['AdSupported'].astype(bool)
data['HaveAds'] = data['HaveAds'].astype(bool)

# Convert 'GenreId' column to categorical data type and then encode it
le = LabelEncoder()
data['GenreId'] = le.fit_transform(data['GenreId'].astype(str))
#data['GenreId'] = data['GenreId'].astype('category')

# Create 'HasScreenshots' and 'HasVideo' columns
data['HasScreenshots'] = data['Screenshots'] > 0
data['HasVideo'] = data['Video'] > 0

# Drop 'Screenshots' and 'Video' columns
#data = data.drop(['Screenshots', 'Video'], axis=1)

# Split the data into features and target variable
X = data.drop('Installs', axis=1)
y = data['Installs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Fit a linear regression model
# model = LinearRegression()

# Fit a RandomForest model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the mean squared error
score = r2_score(y_test, y_pred)

print('R2 score', score)
