import doctest
import scipy.stats as st
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import math

pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("max_columns", None)

df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df.isnull().sum()
df.shape


#####################################
# Zamana Göre Ağırlıklandırılmış Average Rating Hesabı
#####################################

df["overall"].mean()  # Urunun ortalama puani

df.info()
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

a = df["day_diff"].quantile(0.25)
b = df["day_diff"].quantile(0.50)
c = df["day_diff"].quantile(0.75)

df["overall"].mean()


df.loc[df["day_diff"] <= a, "overall"].mean()
df.loc[(df["day_diff"] > a) & (df["day_diff"] <= b), "overall"].mean()
df.loc[(df["day_diff"] > b) & (df["day_diff"] <= c), "overall"].mean()
df.loc[(df["day_diff"] > c), "overall"].mean()


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= a, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > a) & (dataframe["day_diff"] <= b), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > b) & (dataframe["day_diff"] <= c), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > c), "overall"].mean() * w4 / 100


time_based_weighted_average(df)


#####################################
# Ürün için ürün detay sayfasında görüntülenecek 20 review’in belirlenmesi.
#####################################

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


def score_pos_neg_diff(up, down):
    return up-down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up/(up+down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Wilson_Lower_Bound'a göre ilk 20 yorum sıralaması:
df.sort_values(by="wilson_lower_bound", ascending=False).head(20)


#########################################
# Sentiment Analizi (Duygu Analizi)
#########################################

# Yapılan yorumları positive, notr ve negatif olarak sınıflandırmak istiyoruz.
# Bunun için yorumlara duygu analizi yapacağız. Natural Language Toolkit kütüphanesi kullanacağız.

# NLTK -> Doğal dil işleme kütüphanesi (https://www.nltk.org/)
# Sentiment analysis (Duygu analizi), metindeki olumlu veya olumsuz duyguyu tespit etme sürecidir.
# Genellikle şirketler veya markalar tarafından sosyal verilerdeki duyarlılığı tespit etmek, marka itibarını ölçmek
# ve müşterileri anlamak için kullanılır.

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from textblob import TextBlob
nltk.downloader.download("vader_lexicon")

# Daha iyi sonuçlar alabilmek adına, summary değişkenimizin içerisindeki textlerin temizlenmesi gerekiyor
# (Noktalama işaretlerinden kurtulmak, küçük harfe çevirmek...)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
df["summary"] = df["summary"].map(rt)
df["summary"] = df["summary"].str.lower()
df.head(10)

# Sentiment Analizi
# TextBlob çıktı olarak; polarity ve subjectivity değerlerini döndürecektir.
# Polarity; duygu durumunu yani olumlu mu olumsuz mu olduğunu belirtir.
# Bize 0 ile 1 arasında değer döndürür, 1'e ne kadar yakınsa o kadar olumlu, 0'a ne kadar yakınsa o kadar olumsuzdur.

df[["polarity", "subjectivity"]] = df["summary"].apply(lambda text: pd.Series(TextBlob(text).sentiment))

for index, row in df['summary'].iteritems():
     score = SentimentIntensityAnalyzer().polarity_scores(row)
     neg = score['neg']
     neu = score['neu']
     pos = score['pos']
     if neg > pos:
         df.loc[index, 'sentiment'] = "negative"
     elif pos > neg:
         df.loc[index, 'sentiment'] = "positive"
     else:
         df.loc[index, 'sentiment'] = "neutral"
     df.loc[index, 'neg'] = neg
     df.loc[index, 'neu'] = neu
     df.loc[index, 'pos'] = pos


df["sentiment"].value_counts()

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
def create_wordcloud(text):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                  max_words=3000,
                  stopwords=stopwords,
                  repeat=True)
    wc.generate(str(text))
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

create_wordcloud(df[df["sentiment"]=="positive"]["summary"].values)

create_wordcloud(df[df["sentiment"]=="negative"]["summary"].values)
