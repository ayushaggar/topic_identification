from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import string
import os
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


def join_csv(file, df):
    txt_file = 'data/' + file
    df = df.append(
        pd.read_csv(
            txt_file,
            names=[
                'id',
                'date',
                'tweet'],
            encoding='mac_roman',
            skiprows=0,
            sep='|'),
        ignore_index=True)
    return df

# Tokenized and lemmatized document


def clean(doc):
    stop_words = set(stopwords.words('english'))
    stop_punctuations = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join(
        [i for i in doc.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in stop_punctuations)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# removing link in tweet and change to lower case


def prep(df):
    df['tweet'] = df['tweet'].str.replace(r'http\S+|www.\S+', '', case=False)
    df['tweet'] = map(lambda x: x.lower(), df['tweet'])
    df['tweet'] = df['tweet'].apply(lambda x: clean(x).split())
    return df


def main():

    # list down all files in folder
    data_list = os.listdir("data")

    # Handle if no file in data folder
    if len(data_list) == 0:
        print ('no file')
        return

    # defining dataframe for tweets
    col_names = ['id', 'date', 'tweet']
    data_df = pd.DataFrame(columns=col_names)

    # importing data
    for file in data_list:
        data_df = join_csv(file, data_df)

    # cleaning
    data_df = data_df.drop(['id', 'date'], 1)
    data_df = prep(data_df)
    print (data_df)

    processed_docs = []
    for text in list(data_df['tweet']):
        doc = " ".join(text)
        processed_docs.append(doc)

    vectorizer = CountVectorizer(analyzer='word',
                                 # minimum reqd occurences of a word
                                 min_df=12,
                                 stop_words='english',
                                 lowercase=True,
                                 # num chars > 3
                                 token_pattern='[a-zA-Z0-9]{2,}',
                                 )

    data_vectorized = vectorizer.fit_transform(processed_docs)

    # various search params to get best combination
    # n_components is Number of topics.
    search_params = {'n_components': [
        10, 15, 20, 30, 50], 'learning_decay': [.4, .8, .12]}

    # gridsearch to find best parameters
    gsc = GridSearchCV(
        estimator=LatentDirichletAllocation(),
        param_grid=search_params,
        cv=5,
        error_score='numeric')
    gsc.fit(data_vectorized)

    best_lda_model = gsc.best_estimator_

    # model parameters
    print("Best Params: ", gsc.best_params_)

    # log likelihood score
    print("Best Log Likelihood Score: ", gsc.best_score_)

    # Document-Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)
    # column names
    topic_names = [
        "Topic" +
        str(i) for i in range(
            gsc.best_params_.get('n_components'))]
    # index names
    doc_names = ["Doc" + str(i) for i in range(len(processed_docs))]
    # make the pandas dataframe
    df_document_topic = pd.DataFrame(
        np.round(
            lda_output,
            2),
        index=doc_names,
        columns=topic_names)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # Topic Distribution Table
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts(
    ).reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    print (df_topic_distribution)

    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(
        best_lda_model.components_,
        index=topic_names,
        columns=vectorizer.get_feature_names())
    print (df_topic_keywords.head())

main()
