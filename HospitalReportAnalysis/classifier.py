import pandas as pd
from pandas import ExcelWriter
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import unicodedata
import string
import re
import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os.path as op
from sklearn.model_selection  import train_test_split

def train_model(X_train,y_train):
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=True)),('clf', SGDClassifier(alpha=1e-3))])
    text_clf.fit(X_train, y_train)
    return text_clf

def predict(text_clf,X_test,y_test):
    predicted = text_clf.predict(X_test)
    print(np.mean(predicted == y_test))
    print(metrics.classification_report(y_test, predicted))

def pre_process(X,lemma=False):
    X_Preproc = []
    for text in X:
        text_lower = text.lower()
        stop = stopwords.words('english')
        text_nostop = " ".join(filter(lambda word: word not in stop, text_lower.split()))
        if lemma:
            tokens = word_tokenize(text_nostop)
            text_nostop = " ".join([WNlemma.lemmatize(t) for t in tokens])

        X_Preproc.append(text_nostop)
    return X_Preproc


def df_to_list(data_df, training=True):
    data = []
    for row in data_df.iterrows():
        index, value = row
        data.append(value.tolist())

    if training == True:
        #y is causes and X is combination of title cases and Summary
        y = [d[1] for d in data]
        X = [d[2] + ' ' + d[3] for d in data]
    else:
        y = '0'
        if data_df.shape[1] > 2:
            X = [str(d[0]) + ' ' + d[1] + ' ' + d[2] + ' ' + d[3] for d in data]
        else:
            X = [d[1] for d in data]

    return X, y

def load_Data(file_path,evaluate=False):
    if evaluate:
        data_df = pd.read_excel(file_path)
        data_df.columns = ['Cause','CorrectedCause', 'TitleCase', 'SummaryCase']
        data_df[data_df.CorrectedCause == u'Others'] = u'Other'
        categories = data_df['CorrectedCause'].unique()
        categories = [x.encode('UTF8') for x in list(categories)]
        print("%d categories " % len(categories))

    else:
        data_df = pd.read_excel(file_path,header=None)
        data_df.columns = ['Number', 'TitleCase', 'SummaryCase', 'FirstDiagnose', 'Hospitalized']
        del data_df['Number']

    X, y = df_to_list(data_df, training=evaluate)
    X_Preproc = pre_process(X, lemma=False)
    return X_Preproc,y

def evaluate_model():
    file_name = op.join('data','MsiaAccidentCases_Corrected_Final.xlsx')
    X_Preproc, y = load_Data(file_name,evaluate=True)
    X_train, X_test, y_train, y_test = train_test_split(X_Preproc, y, test_size=0.15, random_state=40)
    text_clf = train_model(X_train, y_train)
    predict(text_clf, X_test, y_test)
    return X_Preproc,y

def predict_causes():
    X_train, y_train = evaluate_model()
    file_name = op.join ('data','osha.xlsx')
    X_test, y = load_Data(file_name, evaluate=False)

    text_clf = train_model(X_train, y_train)
    y_pred= text_clf.predict(X_test)
    data_df = pd.read_excel(file_name, header=None)
    data_df.columns = ['Number', 'TitleCase', 'SummaryCase', 'FirstDiagnose', 'Hospitalized']
    data_df['Causes']=y_pred
    file_to_save = op.join('data','osha_processed.csv')
    data_df.to_csv(file_to_save)
    print(y_pred[0:5])

if __name__ == '__main__':
    newstopwords = stopwords.words("English") + ['the', 'is', 'it', 'may']
    WNlemma = nltk.WordNetLemmatizer()
    predict_causes()

