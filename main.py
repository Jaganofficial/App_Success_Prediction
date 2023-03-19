import ast
from datetime import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from wordcloud import WordCloud
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import accuracy_score, r2_score
from sklearn import model_selection  # for splitting into train and test
import json
from urllib.request import urlopen
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


def rmse_score(y1, y2):
    return np.sqrt(np.power(y1 - y2, 2).mean())


#df = pd.read_csv("10000Spartans.csv")#, encoding='unicode_escape')
df = pd.read_csv("3000Spartans_AppData.csv")
print(df.columns)
# HeatMap
# df = df.drop(["Id", "Title", "Description", "GenreId", "ReleasedOn"], axis=1)
# sns.heatmap(df.corr(), cmap='YlGnBu', annot=True, linewidths=0.2)
# plt.show()
# sns.heatmap(df.corr('spearman'), cmap='YlOrBr', annot=True, linewidths=0.2)
# plt.show()

# creating log transformation for reveune
# df['log_Rating'] = np.log1p(df['Rating'])
# fig, ax = plt.subplots()
# plt.hist(df['Rating']);
# plt.title('Distribution of revenue');
# plt.show()
# plt.hist(df['log_Rating']);
# plt.title('Distribution of log transformation of revenue');
# plt.show()


# Review vs installs scaterplot
# df['log_CommentsSentimentalScore'] = np.log1p(df['CommentsSentimentalScore'])
# df['log_Installs'] = np.log1p(df['Installs'])
# plt.scatter(df['log_CommentsSentimentalScore'],df['log_Installs'])
# plt.show()

# installs vs free
# df['log_Installs'] = np.log1p(df['Installs'])
# sns.catplot(x='Free', y='Installs', data=df)
# plt.title('Installs for Apps that are free')
# plt.show()

# words in title and description
# token_title = ' '.join(df['Description'].values) #create split to title by sprace to extract the text.
# #bg color set to white for good contrast, by default bg color is darker
# wordcloud = WordCloud(max_font_size=None, background_color='black', width=1200, height=1000).generate(BeautifulSoup(token_title, "html.parser").text)
# plt.imshow(wordcloud)
# plt.title('Top words from app\'s Descriptions')
# plt.axis("off") # we don't need axes for this
# plt.show()

# installs vs months
# max_installs = df['Installs'].max()
# min_installs = df['Installs'].min()
# df['Installs_normalized'] = (df['Installs'] - min_installs) / (max_installs - min_installs)
# df['Month'] = pd.to_datetime(df['ReleasedOn']).dt.month_name()
# grouped = df.groupby('Month')['Installs'].mean()
# plt.bar(grouped.index, grouped.values)
# plt.xlabel('Month')
# plt.ylabel('Installs')
# plt.title('Installs vs Released Months')
# plt.show()

# installs vs day of a week
# df['DayOfWeek'] = pd.to_datetime(df['ReleasedOn'], format="%b %d, %Y").dt.day_name()
# grouped = df.groupby('DayOfWeek')['Installs'].mean()
# plt.bar(grouped.index, grouped.values)
# plt.xlabel('Day of Week')
# plt.ylabel('Installs')
# plt.show()

# Top genre
# unique_genres = df["GenreId"].apply(pd.Series).stack().unique()
# print("Number of genres: {}".format(len(unique_genres)))
# print("Genres: {}".format(unique_genres))
#
# genres_dummies = pd.get_dummies(df["GenreId"].apply(pd.Series).stack()).sum(level=0) #one hot encoding
# genres_dummies.head()
#
# train_genres = pd.concat([df, genres_dummies],axis=1, sort=False) #merging two data frame
# train_genres.head(5)
#
# genres_overall = train_genres[unique_genres].sum().sort_values(ascending=False)
# plt.figure(figsize=(15,5))
# ax = sns.barplot(x=genres_overall.index, y=genres_overall.values)
# plt.xticks(rotation=90)
# plt.title("Popularity of genres overall")
# plt.ylabel("count")
# plt.show()

# Model Prediction
# selecting the numeric column
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32',
#             'float64']  # so that easy for us to perform  train and test
# df_train = df.select_dtypes(include=numerics)
# df_train = df_train.fillna(df_train.median())
#
# print(df_train.columns)
# data = df_train.copy()
# sns.heatmap(data.corr(), annot=True)
# plt.show()

X = df[['Rating', 'Review', 'Price',
        'Free', 'Sale', 'In_app_purchase', 'Screenshots', 'Video',
        'AdSupported', 'HaveAds', 'CommentsSentimentalScore',
        'CommentReviewValue']]
y = df['Installs']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Linear regression
LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
y_hat = LR_model.predict(X_test)
print("Linear Regression - R Square value: ", r2_score(y_test, y_hat))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.001, max_depth=4, min_samples_split= 2)
gbr.fit(X_train, y_train)
y_hat = gbr.predict(X_test)
print("GradientBoostingRegressor- Test score: ", r2_score(y_test, y_hat))

# Random Forest
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_hat = RF_model.predict(X_test)
print("Random forest- R-Square value:", metrics.r2_score(y_test, y_hat))
