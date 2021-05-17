#pandas code generously referenced from https://www.nicholasrenotte.com/how-to-build-a-sentiment-analyser-for-yelp-reviews-in-python/
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob
from wordcloud import WordCloud

#create the pandas dataframe from the csv file we generated before
df = pd.read_csv('kanka_yelp_reviews.csv')
mpqa = pd.read_csv('mpqa.csv')

#clean up the rating column to only contain numeric values and remove punctuation
extr = df['Rating'].str.extract(r'^(\d)', expand=False)
df['Rating'] = pd.to_numeric(extr)
df['review_lower'] = df['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['review_nopunc'] = df['review_lower'].str.replace('[^\w\s]', '')

#remove stop words, and common words "food" "place" "japanese" etc
stop_words = stopwords.words('english')
df['review_nopunc_nostop'] = df['review_nopunc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
freq= pd.Series(" ".join(df['review_nopunc_nostop']).split()).value_counts()[:30]

#lemmatize all words
df['cleaned_review'] = df['review_nopunc_nostop'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#calculate the sentiment & subjectivity using textblob
df['Polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[0])
df['Subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[1])

#counting positive/negative words from mpqa
conditions = [
    (df['Polarity']<= -.6),
    (df['Polarity'] > -.6) & (df['Polarity']<= -.2),
    (df['Polarity'] > -.2) & (df['Polarity']<= .2),
    (df['Polarity'] > .2) & (df['Polarity']<= .6),
    (df['Polarity']<= 1),
]
values = [1,2,3,4,5]

"""
Assigns an overall sentiment based on the nltk sentiment
1: Negative
2: Somewhat Negative
3: Neutral
4: Somewhat Positive
5: Positive
"""
df['Overall Sentiment'] = np.select(conditions,values)

#Export data to csv file
df.to_csv("analyzed_data.csv", encoding='utf-8', columns=['Review', 'Rating','Polarity','Overall Sentiment'])

print(f"Average Yelp Review: {df['Rating'].mean()}")
print(f"Average Overall Sentiment: {df['Overall Sentiment'].mean()}")

#generating word cloud of all reviews
wordcloud = WordCloud(background_color='white', max_words=200).generate(' '.join(df['cleaned_review']))
wordcloud.to_file('reviewcloud.png')