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


def prep(df, filter_keywords):
    df['tweet'] = df['tweet'].str.replace(r'http\S+|www.\S+', '', case=False)
    df['tweet'] = map(lambda x: x.lower(), df['tweet'])
    if len(filter_keywords) != 0:
        filter_keyword = "|".join(filter_keywords)
        df = df[df['tweet'].str.contains(filter_keyword)]
    df['tweet'] = df['tweet'].apply(lambda x: clean(x).split())
    return df


def train_lda(processed_docs, search_params):
    vectorizer = CountVectorizer(analyzer='word',
                                 # minimum reqd occurences of a word
                                 min_df=12,
                                 stop_words='english',
                                 lowercase=True,
                                 # num chars > 3
                                 token_pattern='[a-zA-Z0-9]{2,}',
                                 )

    data_vectorized = vectorizer.fit_transform(processed_docs)

    # gridsearch to find best parameters
    gsc = GridSearchCV(
        estimator=LatentDirichletAllocation(),
        param_grid=search_params,
        cv=5,
        error_score='numeric')
    gsc.fit(data_vectorized)

    # model parameters
    print("Best Params: ", gsc.best_params_)

    # log likelihood score
    print("Best Log Likelihood Score: ", gsc.best_score_)

    best_lda_model = gsc.best_estimator_

    return [best_lda_model, gsc.best_params_.get('n_components'), vectorizer]


def show_result(best_lda_model, processed_docs, num_of_topics, vectorizer):
    # Document-Topic Matrix
    data_vectorized = vectorizer.fit_transform(processed_docs)
    lda_output = best_lda_model.transform(data_vectorized)
    # column names
    topic_names = [
        "Topic" +
        str(i) for i in range(num_of_topics)]
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
    df_document_topic['doc'] = processed_docs

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

    # Top 10 keywords for each topic
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in best_lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:10]
        topic_keywords.append(keywords.take(top_keyword_locs))

    # Topic - Top 10 Keywords Dataframe
    df_topic_top_keywords = pd.DataFrame(topic_keywords)
    df_topic_top_keywords.columns = [
        'Word ' +
        str(i) for i in range(
            df_topic_top_keywords.shape[1])]
    df_topic_top_keywords.index = [
        'Topic ' +
        str(i) for i in range(
            df_topic_top_keywords.shape[0])]
    print (df_topic_top_keywords)

    return df_document_topic


def main(filter_keywords):
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
    data_df = prep(data_df, filter_keywords)

    data_docs = list(data_df['tweet'])
    processed_docs = []
    for text in data_docs:
        doc = " ".join(text)
        processed_docs.append(doc)

    # various search params to get best combination
    # n_components is Number of topics.
    search_params = {'n_components': [
        6, 10, 20], 'learning_decay': [.8, .12]}

    [best_lda_model, num_of_topics, vectorizer] = train_lda(
        processed_docs, search_params)

    df_document_topic = show_result(
        best_lda_model,
        processed_docs,
        num_of_topics,
        vectorizer)

    # Exporting Model
    file_name = 'lda_model'
    pickle.dump(best_lda_model, open(file_name, 'w'))

    return [df_document_topic, num_of_topics]
