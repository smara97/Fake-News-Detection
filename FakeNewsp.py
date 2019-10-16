import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
%matplotlib inline


trainingset=pd.read_csv('../input/fake-news/train.csv')
testingset=pd.read_csv('../input/fake-news/test.csv')


trainingset.head(10)


authors={}
for i in range(trainingset.shape[0]):
    if type(trainingset.loc[i,'author'])!=float:
        if trainingset.loc[i,'author'] in authors:
            if trainingset.loc[i,'label']==1:
                authors[trainingset.loc[i,'author']]=[authors[trainingset.loc[i,'author']][0]+1,authors[trainingset.loc[i,'author']][1]+1,authors[trainingset.loc[i,'author']][2]]
            else:
                authors[trainingset.loc[i,'author']]=[authors[trainingset.loc[i,'author']][0]+1,authors[trainingset.loc[i,'author']][1],authors[trainingset.loc[i,'author']][2]+1]
        else:
            if trainingset.loc[i,'label']==1:
                authors[trainingset.loc[i,'author']]=[1,1,0]
            else:
                authors[trainingset.loc[i,'author']]=[1,0,1]
    else:
        trainingset.loc[i,'author']="Unknown"
        
print(len(authors))


print(trainingset.shape)
trainingset.head(10)


testingset['label']='t'

testingset=testingset.fillna(' ')
trainingset=trainingset.fillna(' ')
testingset['total']=testingset['title']+' '+testingset['author']+testingset['text']
trainingset['total']=trainingset['title']+' '+trainingset['author']+trainingset['text']

transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 3))
counts = count_vectorizer.fit_transform(trainingset['total'].values)
tfidf = transformer.fit_transform(counts)


targets = trainingset['label'].values
test_counts = count_vectorizer.transform(testingset['total'].values)
test_tfidf = transformer.fit_transform(test_counts)

X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
print('Accuracy of Lasso classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Lasso classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))

title="Obama is egyptian."
author="Smara"
text="Obama not american citizen."
tst=pd.DataFrame()
tst.loc[0,'total']=title+' '+author+text
tst.head()
example_counts = count_vectorizer.transform(tst['total'].values)

predictions = logreg.predict(example_counts)

if predictions[0]==1:
    print("Fake")
else:
    print("True")