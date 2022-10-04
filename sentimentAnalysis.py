from matplotlib.pyplot import title
from requests_html import HTMLSession
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re, string
import csv
import random

from torch import neg, positive

positive_news = []
negative_news = []
neutral_news = []

#OPENS AND READS DATA, SEPARATES INTO RESPECTIVE LIST
with open("financial_data1.csv", "r") as infile:
    reader = csv.reader(infile, delimiter= ",")
    for row in reader:
        if(row[1]) == "positive":
            if(row[0].lower() not in positive_news):
                positive_news.append(row[0].lower())
        if(row[1]) == "negative":
            if(row[0].lower() not in negative_news):
                negative_news.append(row[0].lower())
        if(row[1]) == "neutral":
            if(row[0].lower() not in negative_news):
                neutral_news.append(row[0].lower())

with open("financial_data2.csv", "r", encoding = "ISO-8859-1") as infile:
    reader = csv.reader(infile, delimiter= ",")
    for row in reader:
        if(row[0]) == "positive":
            if(row[1].lower() not in positive_news):
                positive_news.append(row[1].lower())
        if(row[0]) == "negative":
            if(row[1].lower() not in negative_news):
                negative_news.append(row[1].lower())
        if(row[0]) == "neutral":
            if(row[1].lower() not in neutral_news):
                neutral_news.append(row[1].lower())

with open('Sentences_50Agree.txt', encoding="ISO-8859-1") as f:
    for line in f.readlines():
        currNews = line[:line.index("@")]
        if "positive" in line:
            if(currNews not in positive_news):
                positive_news.append(currNews.lower())
        if "negative" in line:
            if(currNews not in negative_news):
                negative_news.append(currNews.lower())
        if "neutral" in line:
            if(currNews not in neutral_news):
                neutral_news.append(currNews.lower())

#FILTERING AND CLEANING THE DATA
def clean(tokenList, stop_words = ()): 

    cleaned_tokens = []

    for token, tag in pos_tag(tokenList):

        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token not in stop_words:
            if token.startswith("http") or token.startswith("mn") or token.startswith("'s") or token.startswith("``") or token.startswith("eur") or token.startswith(".") or token.startswith("//") or token.startswith("x"):
                pass
            else:
                cleaned_tokens.append(token)

            
    return cleaned_tokens

stopWords = stopwords.words('english')

positivefd = (nltk.word_tokenize(''.join(positive_news)))
lemmatizedPositiveFD = clean(positivefd, stopWords)

negativefd  = (nltk.word_tokenize(''.join(negative_news)))
lemmatizedNegativeFD = clean(negativefd, stopWords)  

neutralfd  = (nltk.word_tokenize(''.join(neutral_news)))
lemmatizedNeutralFD = clean(neutralfd, stopWords)  

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([tweet_tokens, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(lemmatizedPositiveFD)
negative_tokens_for_model = get_tweets_for_model(lemmatizedNegativeFD)
neutral_tokens_for_model = get_tweets_for_model(lemmatizedNeutralFD)

positive_dataset = [(tweet_dict, "Positive")
    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
    for tweet_dict in negative_tokens_for_model]

neutral_dataset = [(tweet_dict, "Neutral")
    for tweet_dict in neutral_tokens_for_model]

dataset = positive_dataset + negative_dataset + neutral_dataset

random.shuffle(dataset)

train_data = dataset[:50000]
test_data = dataset[4000:]

classifier = NaiveBayesClassifier.train(train_data)

#print("Accuracy is:", classify.accuracy(classifier, test_data))
# print(classifier.show_most_informative_features(30))

#SCRAPING DATA FROM GOOGLE NEWS
numPositive = 0
numNegative = 0
session = HTMLSession()
URL = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB/sections/CAQiXENCQVNQd29JTDIwdk1EbHpNV1lTQW1WdUdnSlZVeUlQQ0FRYUN3b0pMMjB2TURsNU5IQnRLaG9LR0FvVVRVRlNTMFZVVTE5VFJVTlVTVTlPWDA1QlRVVWdBU2dBKioIAComCAoiIENCQVNFZ29JTDIwdk1EbHpNV1lTQW1WdUdnSlZVeWdBUAFQAQ?hl=en-US&gl=US&ceid=US%3Aen"
r = session.get(URL)
r.html.render(sleep = 1, scrolldown = 5)
articles = r.html.find('article')
for item in articles:
    try:
        newsitem = item.find('h3', first=True) 
        title = newsitem.text
        if(classifier.classify(dict([token, True] for token in title)) == "Positive"):
            print(title + ": " + "positive")
            numPositive = numPositive + 1
        elif (classifier.classify(dict([token, True] for token in title)) == "Negative"):
            print(title + ": " + "negative")
            numNegative = numNegative + 1
    except:
        pass

if(numPositive>numNegative):
    print("Predicted market movement to be up")
elif(numNegative>numPositive):
    print("Predicted market movement to be down")
else:
    print("Unable to predict market movement")

#VADER
cs = []
analyzer = SentimentIntensityAnalyzer()
for item in articles:
    cs.append(analyzer.polarity_scores(str(item))['compound'])

print(cs)