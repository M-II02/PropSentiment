
import csv
import re
import string
import preprocessor
import preprocessor as p
import string
import datetime
import numpy as np
import pandas as pd

!pip install snscrape
import snscrape.modules.twitter as sntwitter
!pip install vaderSentiment
!pip install neattext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
!pip install tweet-preprocessor

from PIL import Image
import plotly.express as px
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.stem import WordNetLemmatizer 
from matplotlib import pyplot as plt

import seaborn as sn
np.random.seed(1)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle

query = "(#airline) until:2022-10-11 since:2022-09-01"
tweets = []
limit = 50


for tweet in sntwitter.TwitterHashtagScraper(query).get_items():
    
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.url, tweet.content])

df = pd.DataFrame(tweets, columns=['Date', 'TweetURL','Tweet'])

df.to_csv('extracted.csv')

df.head(50)

'''def cleanTxt(Tweet):
    text = re.sub('@[A-Za-z0–9]+', '', Tweet) #Removing @mentions
    text = re.sub('#', '', Tweet) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', Tweet) # Removing RT
    text = re.sub('https?:\/\/\S+', '', Tweet) # Removing hyperlink
    return text

#applying this function to Text column of our dataframe
df["Tweet"] = df["Tweet"].apply(cleanTxt)'''

def clean_text(Tweet):
    #Remove hyper links
    text = re.sub(r'https?:\/\/\S+', ' ', Tweet)
    
    #Remove @mentions
    text = re.sub(r'@[A-Za-z0-9]+', ' ', Tweet)
    
    #Remove anything that isn't a letter, number, or one of      the punctuation marks listed
    text = re.sub(r"[^A-Za-z0-9#'?!,.]+", ' ', Tweet)   
    
    return text

df['Tweet'] = df['Tweet'].apply(clean_text)

df['Tweet'] = df['Tweet'].str.lower()

'''df= df.drop_duplicates()
df['clean_Tweet']=df['Tweet'].str.lower()

df['clean_Tweet'] = df['Tweet'].apply(lambda x: re.sub(r"http\S+","",x))
df['clean_Tweet']=df['Tweet'].str.replace(r'\@\S+'," ",regex=True)
df['clean_Tweet']=df['Tweet'].str.replace(r'#\S+'," ",regex=True)
df['clean_Tweet']=df['Tweet'].str.replace(r'\$\S+'," ",regex=True)
df['clean_Tweet']=df['Tweet'].str.replace(r'\$\S+'," ",regex=True)
df['clean_Tweet']=df['Tweet'].apply(lambda x:[w for w in x if len(w) >= 3])
df['clean_Tweet']=df['Tweet'].apply(lambda x:" ".join(x))'''

df[['Tweet']].head(20)

SIA = SentimentIntensityAnalyzer()
df['Tweet']=df["Tweet"].astype(str)

df['Polarity']=df["Tweet"].apply(lambda x: SIA.polarity_scores(x)['compound'])
df['Neutral Score']=df["Tweet"].apply(lambda x: SIA.polarity_scores(x)['neu'])
df['Negative Score'] = df["Tweet"].apply(lambda x: SIA.polarity_scores(x)['neg'])
df['Positive Score'] = df["Tweet"].apply(lambda x: SIA.polarity_scores(x)['pos'])

df['Sentiment']= ''
df.loc[df['Polarity'] > 0, 'Sentiment']="Positive"
df.loc[df['Polarity']== 0,'Sentiment']="Neutral"
df.loc[df['Polarity'] < 0 ,'Sentiment']="Negative"

df.head(20)

df.to_csv('extractedS.csv')

df_ = df[['Sentiment','Tweet']].copy()

df_['Sentiment'].hist()

target_map = {'Positive':1,'Negative':0, 'Neutral':2}
df_['target'] = df_['Sentiment'].map(target_map)

df_.head(25)

dftrain, dftest = train_test_split(df_)

dftrain.head()

vectorizer = TfidfVectorizer(max_features=2000)

X_train = vectorizer.fit_transform(dftrain['Tweet'])

X_train

X_test = vectorizer.transform(dftest['Tweet'])

Y_train = dftrain['target']
Y_test = dftest['target']

df['Sentiment'].value_counts()

model = LogisticRegression(max_iter =500)
model.fit(X_train,Y_train)
print("Train acc:",model.score(X_train,Y_train))
print("Test acc:",model.score(X_test,Y_test))

Pr_train = model.predict_proba(X_train)#[:, 1]
Pr_test = model.predict_proba(X_test)#[:, 1]
print("Train AUC", roc_auc_score(Y_train,Pr_train,multi_class ='ovo'))
print("Test AUC", roc_auc_score(Y_test,Pr_test,multi_class ='ovo'))

P_train = model.predict(X_train)
P_test = model.predict(X_test)

cm = confusion_matrix(Y_train,P_train, normalize = "true")
cm

import pandas as pd

def plot_cm(cm):
  classes = ['Negative','Positive','Neutral']
  df_cm =pd.DataFrame(cm, index=classes, columns=classes)
  ax =sn.heatmap(df_cm,annot=True, fmt ='g')
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Target")

plot_cm(cm)

cm_test =confusion_matrix(Y_test,P_test, normalize='true')
plot_cm(cm_test)

df['Sentiment'].value_counts()

Tweets = df.groupby(['Sentiment']).size().reset_index(name='Counts')
Tweets

binary_target_list = [target_map['Positive'],target_map['Negative']]
dfb_train= dftrain[dftrain['target'].isin(binary_target_list)]
dfb_test= dftest[dftest['target'].isin(binary_target_list)]

dfb_train.head()

X_train= vectorizer.fit_transform(dfb_train['Tweet'])
X_test = vectorizer.transform(dfb_test['Tweet'])

Y_train = dfb_train['target']
Y_test = dfb_test['target']

model=LogisticRegression(max_iter=500)
model.fit(X_train,Y_train)
print("Train acc:",model.score(X_train,Y_train))
print("Test acc:",model.score(X_test,Y_test))

Pr_train=model.predict_proba(X_train)[:,-1]
Pr_test=model.predict_proba(X_test)[:,-1]
print("Train AUC:",roc_auc_score(Y_train,Pr_train))
print("Test AUC:",roc_auc_score(Y_test,Pr_test))

model.coef_

plt.hist(model.coef_[0],bins=30);

word_index_map = vectorizer.vocabulary_
word_index_map

threshold=0.2

print("Most positive words:")
for word, index in word_index_map.items():
  weight = model.coef_[0][index]
  if weight > threshold:
    print(word,weight)

print("Most negative words:")
for word,index in word_index_map.items():
  weight = model.coef_[0][index]
  if weight < -threshold:
    print(word,weight)

import pandas as pd
from matplotlib import pyplot as plot_cm

plt.rcdefaults()
plt.style.use('fivethirtyeight')

data = pd.read_csv('extractedS.csv')

date_ = data['Date']

Neutral_Score = data['Neutral Score']	
Negative_Score = data['Negative Score']	
Positive_Score = data['Positive Score']

plt.plot(date_, Neutral_Score, label="Neutral Score")
plt.plot(date_, Negative_Score, label="Negative Score")
plt.plot(date_, Positive_Score, label="Positive Score")

plt.legend()
plt.title('Sentiment Analysis')
plt.ylabel('Sentiment')
plt.xlabel('Date')

plt.savefig("plot.png")
plt.show()

from matplotlib import pyplot as plt
import pandas as pd

plt.style.use('bmh')
df = pd.read_csv('extractedS.csv')

x= df['Sentiment']
y = df['Date']

plt.xlabel('Sentiment',fontsize=18)
plt.ylabel('Date',fontsize=16)
#plt.pie(y,labels=x, radius=1.2,autopct='0.01f%%',shadow=True,explode=[.05,.2,.05,.2,.05,.2,.05])
plt.plot(x,y)
plt.show()

#  N E W  #

#https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.stem import WordNetLemmatizer 
from matplotlib import pyplot as plt

import seaborn as sn
np.random.seed(1)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

df_ = pd.read_csv('airline_sentiment.csv')

df_.head()

df = df_[['airline_sentiment','text']].copy()

df.head(20)

df['airline_sentiment'].hist()

target_map = {'positive':1,'negative':0, 'neutral':2}
df['target'] = df['airline_sentiment'].map(target_map)

df.head()

df_train, df_test = train_test_split(df)

df_train.head()

vectorizer = TfidfVectorizer(max_features=2000)

X_train = vectorizer.fit_transform(df_train['text'])

X_train

X_test = vectorizer.transform(df_test['text'])

Y_train = df_train['target']
Y_test = df_test['target']

df['airline_sentiment'].value_counts()

'''from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(target)

print(utils.multiclass.type_of_target(target))

print(utils.multiclass.type_of_target(target.astype('int')))

print(utils.multiclass.type_of_target(target))'''

model = LogisticRegression(max_iter =1000)
model.fit(X_train,Y_train)
print("Train acc:",model.score(X_train,Y_train))
print("Test acc:",model.score(X_test,Y_test))

Pr_train = model.predict_proba(X_train)#[:, 1]
Pr_test = model.predict_proba(X_test)#[:, 1]
print("Train AUC", roc_auc_score(Y_train,Pr_train,multi_class ='ovo'))
print("Test AUC", roc_auc_score(Y_test,Pr_test,multi_class ='ovo'))

P_train = model.predict(X_train)
P_test = model.predict(X_test)

cm = confusion_matrix(Y_train,P_train, normalize = "true")
cm

import pandas as pd

def plot_cm(cm):
  classes = ['negative','positive','neutral']
  df_cm =pd.DataFrame(cm, index=classes, columns=classes)
  ax =sn.heatmap(df_cm,annot=True, fmt ='g')
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Target")

plot_cm(cm)

cm_test =confusion_matrix(Y_test,P_test, normalize='true')
plot_cm(cm_test)

df['airline_sentiment'].value_counts()

Tweets = df.groupby(['airline_sentiment']).size().reset_index(name='Counts')
Tweets

'''import re

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

emoticons = emoticons_happy.union(emoticons_sad)
'''

'''import string
from collections import Counter

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

text =  open('extracted.csv',encoding='utf-8').read()
lower_case = text.lower()
cleaned_text= lower_case.translate(str.maketrans('','',string.punctuation))

tokenized_words = cleaned_text.split()


stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

final_words = []
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)

print(final_words)'''

'''import os
import sys

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_emotion_analysis.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)'''

import pickle

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

#Model optimization using GridSearchCV

parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
model = SVC()
model_grid = GridSearchCV(model,parameters, cv=5)

model_grid.fit(X_train,Y_train)

print(model_grid.best_params_)
print(model_grid.best_estimator_)

filename = 'sentiment_analysis.pkl'

pickle.dump(model_grid, open(filename, 'wb'))

#Load and test saved model
#loaded_model = pickle.load(open(filename, 'rb'))

#review = ["This is pretty much the worse movie I have ever watched. It's completely thrash!"]
#new_review = vectorizer.transform(review)

#result = loaded_model.predict(new_review)
#print(result)
