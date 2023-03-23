import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns  # plotting
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

apps_data = pd.read_csv("3000Spartans_AppData.csv")  # App dataset extracted from playstore

# PreProcessing
# Dropping the columns with null values:
apps_data = apps_data.dropna()
apps_data['Installs'] = apps_data['Installs'].interpolate()
apps_data['Rating'] = apps_data['Rating'].interpolate()
apps_data['Review'] = apps_data['Review'].interpolate()

print("First 5 rows of the dataset: \n", apps_data.head())

# Columns in our dataset
# ------------------------------------------------------------------------------------#
# 'Id', 'Title', 'Description', 'Installs', 'Rating', 'Review', 'Price',
# 'Free', 'Sale', 'In_app_purchase', 'GenreId', 'Screenshots', 'Video',
# 'AdSupported', 'HaveAds', 'ReleasedOn', 'CommentsSentimentalScore',
# 'CommentReviewValue' - 18
print(apps_data.columns, end="\n")

# App categories in our dataset
# ------------------------------------------------------------------------------------#
# 'BUSINESS', 'EDUCATION', 'PARENTING', 'SOCIAL', 'LIFESTYLE',
# 'PRODUCTIVITY', 'COMMUNICATION', 'TRAVEL_AND_LOCAL', 'FINANCE',
# 'BOOKS_AND_REFERENCE', 'HEALTH_AND_FITNESS', 'TOOLS', 'ENTERTAINMENT',
# 'HOUSE_AND_HOME', 'FOOD_AND_DRINK', 'MEDICAL', 'NEWS_AND_MAGAZINES',
# 'PERSONALIZATION', 'MUSIC_AND_AUDIO', 'WEATHER', 'VIDEO_PLAYERS', 'PHOTOGRAPHY',
# 'SHOPPING', 'ART_AND_DESIGN', 'MAPS_AND_NAVIGATION', 'AUTO_AND_VEHICLES', 'SPORTS',
# 'LIBRARIES_AND_DEMO', 'BEAUTY', 'DATING', 'EVENTS', 'COMICS' - 32
categories = list(apps_data["GenreId"].unique())
print("There are {0:.0f} categories!".format(len(categories)), end="\n")
print(categories, end="\n")

# Exploration
print(apps_data.info())
print(apps_data['Installs'].describe(), end="\n")
print(apps_data['Rating'].describe(), end="\n")
print(apps_data['Review'].describe(), end="\n")

# Ratings in our dataset
apps_data['Rating'].plot(kind='hist')
plt.show()

# Top geners in our dataset
plt.figure(figsize=(15, 5))
top_genres = apps_data["GenreId"].value_counts().index
sns.countplot(data=apps_data, x='GenreId', order=top_genres)
plt.xticks(rotation=75)
plt.title('Top genres in our dataset')
plt.xlabel('Genres')
plt.ylabel('No. of Apps')
plt.show()
le = LabelEncoder()

# fit and transform the 'genres' column
apps_data['GenreId'] = le.fit_transform(apps_data['GenreId'])
# Analysis
final_app_data = apps_data.drop(['Id','Title', 'Description','ReleasedOn'],axis=1)
print("There are {} total rows.".format(final_app_data.shape[0]))
countPop = final_app_data[final_app_data["Installs"] > 100000].count()
print("{} Apps are Popular!".format(countPop[0]))
print("{} Apps are Unpopular!\n".format((final_app_data.shape[0]-countPop)[0]))

print("For an 80-20 training/test split, we need about {} apps for testing\n".format(final_app_data.shape[0]*.20))

# condition to filter rows
condition = final_app_data['Installs'] > 100000
print(condition)
# update the 'name' column for the rows that meet the condition
final_app_data['Installs'] = np.where(condition, 1, 0)

X = final_app_data.drop(['Installs'], axis=1)
y = final_app_data['Installs']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = model.predict(X_test)
# Calculate the mean squared error
score = accuracy_score(y_test, y_pred)
print(score)

popularity_classifier = DecisionTreeClassifier(max_leaf_nodes=29, random_state=0)
popularity_classifier.fit(X_train, y_train)
y_pred = popularity_classifier.predict(X_test)
print(accuracy_score(y_true = y_test, y_pred = y_pred))
