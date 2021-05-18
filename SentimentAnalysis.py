#some pandas code referenced from https://www.nicholasrenotte.com/how-to-build-a-sentiment-analyser-for-yelp-reviews-in-python/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob
from wordcloud import WordCloud

#create the pandas dataframe from the csv file we generated before
df = pd.read_csv('kanka_yelp_reviews.csv')

#clean up the rating column to only contain numeric values and remove punctuation
extr = df['Rating'].str.extract(r'^(\d)', expand=False)
df['Rating'] = pd.to_numeric(extr)
df['cleaned_review'] = df['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['cleaned_review'] = df['cleaned_review'].str.replace('[^\w\s]', '')

#remove stop words, and common words "food" "place" "japanese" etc
stop_words = stopwords.words("english")
df['cleaned_review'] = df['cleaned_review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
other_stop_words = ['place','food']
df['cleaned_review'] = df['cleaned_review'].apply(lambda x: " ".join(x for x in x.split() if x not in other_stop_words))
#lemmatize all words
df['cleaned_review'] = df['cleaned_review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

predictions = [
    (df['Rating'] == 1),
    (df['Rating'] == 2),
    (df['Rating'] == 3),
    (df['Rating'] == 4),
    (df['Rating'] == 5),
]
pvalues = [-1, -.5, 0, .5, 1]

#calculate the sentiment & subjectivity using textblob
df['Predicted Sentiment'] = np.select(predictions, pvalues)
df['Sentiment'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[0])
df['Subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[1])

#counting positive/negative words from mpqa
conditions = [
    (df['Sentiment']<= -.6),
    (df['Sentiment'] > -.6) & (df['Sentiment']<= -.2),
    (df['Sentiment'] > -.2) & (df['Sentiment']<= .2),
    (df['Sentiment'] > .2) & (df['Sentiment']<= .6),
    (df['Sentiment']<= 1),
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
df.to_csv("analyzed_data.csv", encoding='utf-8', columns=['cleaned_review', 'Rating','Predicted Sentiment', 'Sentiment','Overall Sentiment', 'Subjectivity'])

#Generate Average Values for the Data
print(f"Average Yelp Review: {df['Rating'].mean()}")
print(f"Average Overall Sentiment: {df['Overall Sentiment'].mean()}")
print(f"Average Sentiment: {df['Sentiment'].mean()}")
print(f"Average Subjectivity: {df['Subjectivity'].mean()}")

#generating word cloud of all reviews
wordcloud = WordCloud(background_color='white', max_words=200).generate(' '.join(df['cleaned_review']))
wordcloud.to_file('reviewcloud.png')

#create a word frequency diagram for the ten most common words in our cleaned data set
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title("Word Frequency")
pd.Series(" ".join(df['cleaned_review']).split()).value_counts()[:10].plot(kind="bar", color = "green")
plt.savefig('word_frequency.png')

#find the most common words in the data set and their overall sentiment
freq = pd.Series(" ".join(df['cleaned_review']).split()).value_counts()[:1000].index.tolist()
word_sent = []
for x in freq:
    sent = TextBlob(x).sentiment[0]
    word_sent.append([x,sent])
sent = [x for x in word_sent if x[1] != 0.0]

#Calculate the average sentiment, and plot the neg/pos sentiments
pos_sent = [x for x in sent if x[1] > 0.0]
neg_sent = [x for x in sent if x[1] < 0.0]

#I was not able to parse the lists properly to do this average in one line
#First we are calculating the sum and dividing by the length of the list afterwards, giving us the average
#sentiment of the positive and negative words.
sum_pos = 0
sum_neg = 0
pos_words = []
neg_words = []
for pos in pos_sent:
    sum_pos += pos[1]
    pos_words.append(pos[0])
for neg in neg_sent:
    sum_neg += neg[1]
    neg_words.append(neg[0])
avg_pos = sum_pos/len(pos_sent)
avg_neg = sum_neg/len(neg_sent)

print(f'Average positive sentiment: {avg_pos} \nAverage negative sentiment: {avg_neg}')

#Generate a word cloud based on the negative and positive words in the reviews
word_list = ' '.join([i for i in df['cleaned_review']]).split()
wordcloud_neg = WordCloud(background_color='black', max_words=200)
wordcloud_pos = WordCloud(background_color='black', max_words=200)

wc_neg =[]
wc_pos = []

#create a list of any occurences of positive or negative words in our word list
#allows us to create a positive and negative word cloud allowing us to visualize the most common positive/negative words
for word in word_list:
    if word in pos_words:
        wc_pos.append(word)
    if word in neg_words:
        wc_neg.append(word)

#appends the list to a string allowing us to generate the wordclouds
wcn = ' '.join(wc_neg)
wcp = ' '.join(wc_pos)
wordcloud_neg.generate(wcn)
wordcloud_pos.generate(wcp)
wordcloud_neg.to_file('negative_words.png')
wordcloud_pos.to_file('positive_words.png')