---
layout: post
title:  "Melbourne Datathon 2016 - 4th Place"
date:   2016-05-20 16:23:02 +1000
categories: hackathon
---

![](C:\Users\Jared Chung\Desktop\JaredChung.github.io\_posts\Kaggle_Image.PNG)

The Melbourne Datathon 2016 is a hackathon which was organised by the Melbourne data science meet up group as way to bring data scientists together by solving real world problems. The event started with an introductory night on the 21st of April with sneak peak at the dataset, followed by a full hack day on the 23rd which was held in the Telstra Gurrowa Innovation Lab.
​	
The dataset contained job ads extracted from the <a href="www.seek.com.au">Seek</a> website which is used by companies for recruitment. The competition was broken in two parts, the first part was a challenge to extract interesting insights from the data where the top 5 teams would present their results.

The second part was a predictive modelling competition which was hosted on the data science website Kaggle. The objective was to classify job ads as "Hospitality and Tourism" class or not using the data provided as well as additional external sources.  

I decided to compete in the second part of the competition as I wanted to apply some of the machine learning techniques I had recently learnt.

## Part 1 - Data Cleaning 

As this was a natural language processing (NLP) problem, I spent a large amount of time scanning through the text data and identifying misspellings and redundancy in text for example, "full time permanent" is the same as "full time". Below is in exert from the code which shows the extent of the cleaning used in this competition.

```python
# Cleaning Function
def clean_raw_job_type(s):
    s = s.lower()
    s = s.replace("  "," ")
    s = re.sub("[^a-zA-Z]", " ",s)
    s = s.replace("  "," ")
    s = s.replace("   "," ")
    s = s.replace("    "," ")
    s = s.replace("  "," ")
    s = re.sub(" [a-z] ","",s)
    s = s.rstrip()
    s = s.lstrip()
    
    # Full Time Clean
    s = s.replace("fulltime","full time")
    s = s.replace("full time permanent","full time")
    s = s.replace("permanent full time","full time")
    s = s.replace("full time temporary contract","full time")
    s = s.replace("full time position","full time")
    s = s.replace("full time experienced","full time")
    s = s.replace("worktype full time","full time")
    s = s.replace("work type full time","full time")
    s = s.replace("full time regular","full time")
    s = s.replace("full time flexible","full time")
    
    words = s.split()                                             
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   

    return " ".join(meaningful_words)
```
## Part 2 - Data Processing

The second part of the competition was spent transforming the newly processed data. One of the processing techniques for natural language processing is to apply Stemming or Lemmatization. The purpose of both of these techniques is to reduce the words into their root form. After testing out both techniques, I found Lemmatization to work slightly better then stemming.

```
# Lemma is the best out of porter and snowball

lemma = WordNetLemmatizer()
stops = set(stopwords.words("english"))

def lemma_tokens(tokens, lemma):
    lemmatized = []
    for item in tokens:
        lemmatized.append(lemma.lemmatize(item))
    return lemmatized
```

Now the text has been cleaned and simplified, it still isn't ready, this is because machine learning models can't process raw text directly. The strategy then is to convert the text into numbers using a technique called Bag of Words. 


```python
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),min_df=2,use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfidf_vectorizer.fit(train_X)
train_X_tfidf =  tfidf_vectorizer.transform(train_X)
test_X_tfidf = tfidf_vectorizer.transform(test_X)
```


## Part 3 - Modelling

```
from sklearn.linear_model import LogisticRegressionCV
logreg = LogisticRegressionCV(class_weight = 'balanced',cv = 5,scoring='roc_auc',n_jobs=-1,random_state=42)
logreg.fit(train_X_tsvd, train_Y)
print "Score %.5f +/- %.5f " % (logreg.scores_[1].mean(),logreg.scores_[1].std())
```



## Part 4 - Conclusion

