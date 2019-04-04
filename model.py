import os
import pandas as pd

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')
# pd.set_option('display.max_colwidth', -1)

def join_csv(file, df):
    txt_file = 'data/' + file
    df = df.append(pd.read_csv(txt_file, names=['id','date','tweet'], skiprows=0, sep='|'), ignore_index = True)
    return df

def clean(doc):
    stop_words = set(stopwords.words('english'))
    stop_punctuations = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in stop_punctuations)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# removing link in tweet and change to upper case
def prep(df):
    df['tweet'] = df['tweet'].str.replace('http\S+|www.\S+', '', case=False)
    df['tweet'] = map(lambda x: x.upper(),df['tweet'])
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
    col_names =  ['id','date','tweet']
    data_df = pd.DataFrame(columns = col_names)

    # importing data
    for file in data_list:
        data_df = join_csv(file, data_df)

    # cleaning 
    data_df = data_df.drop(['id','date'], 1)
    data_df = prep(data_df)
    print (data_df)

main()