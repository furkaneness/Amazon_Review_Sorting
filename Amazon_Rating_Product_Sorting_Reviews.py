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

df["day_diff"].quantile([.1, .25, .60])

df.loc[df["day_diff"] < 167, "overall"].mean() * 35/100 +\
df.loc[(df["day_diff"] > 167) & (df["day_diff"] < 281), "overall"].mean() * 30/100 +\
df.loc[(df["day_diff"] > 281) & (df["day_diff"] < 497), "overall"].mean() * 20/100 +\
df.loc[df["day_diff"] > 497, "overall"].mean() * 15/100


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



