---
layout: post
title:  "Melbourne Datathon 2016 - 4th Place"
date:   2016-05-20 16:23:02 +1000
categories: hackathon
---

The Melbourne Datathon 2016 is a hackathon which was organised by the Melbourne data science meet up group as way to bring data scientists together by solving real world problems. The event started with an introductory night on the 21st of April with sneak peak at the dataset, followed by a full hack day on the 23rd which was held in the Telstra Gurrowa Innovation Lab.
​	
The dataset contained job ads extracted from the <a href="www.seek.com.au">Seek</a> website which is used by companies for recruitment. The competition was broken in two parts, the first part was a challenge to extract interesting insights from the data where the top 5 teams would present their results.

The second part was a predictive modelling competition which was hosted on the data science website Kaggle. The objective was to classify job ads as "Hospitality and Tourism" class or not using the data provided as well as additional external sources.  

I decided to compete in the second part of the competition as I wanted to apply some of the machine learning techniques had recently acquired.

As this was a natural language processing (NLP) problem, I spent a large amount of time scanning through the text data and identifying misspellings and redundancy in text for example, "full time permanent" is the same as "full time". Below is in exert from my code which shows the extent of the cleaning used in this competition.

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
    
    
    # Part Time Clean
    s = s.replace("part time permanent","part time")
    s = s.replace("part time position","part time")
    s = s.replace("permanent part time","part time")
    
    
    # Casual Clean
    s = s.replace("casual vacation","casual")
    s = s.replace("daily hourly rate casual","casual")
    s = s.replace("casual position","casual")
    s = s.replace("daily hourly rate","casual")
    s = s.replace("worktype casual","casual")
    
    # Temp
    s = s.replace("contract temp","temp")
    
    words = s.split()                                             
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   

    return " ".join(meaningful_words)
```
In addition to the cleaning of the text, an important 

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






```python
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),min_df=2,use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfidf_vectorizer.fit(train_X)
train_X_tfidf =  tfidf_vectorizer.transform(train_X)
test_X_tfidf = tfidf_vectorizer.transform(test_X)
```


```

```

