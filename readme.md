## Objective
> Topic and Sub-topic Identification from Health News Data

**** 
> Topic detection/clustering :

**Output** :
1) Representative keywords for each cluster
2) Topic Vs Number of Documents
3) Accuracy
4) Best Parameter of the model
5) Model

> Sub Topic detection/clustering :

**Output**
1) Representative keywords for each cluster
2) Topic Vs Number of Documents
3) Accuracy
4) Best Parameter of the model
5) Model

**Note**: Output is shown as print

**** 

## Tools use 
> Python 2.7

> Main Libraries Used -
1) nltk
2) numpy
3) pandas
4) scikit-learn

**** 

## Installing

```sh
$ git clone https://github.com/ayushaggar/topic_identification.git
$ cd topic_identification
$ pip install -r requirements.txt
``` 
For topic - Task 1
```sh
$ python topic.py
```
For Sub Topic - Task 2
```sh
$ python subtopic.py
```
****
## Various Steps in approach are -

    1) Text processing techniques used 
    Lemmatizing -
    It is process of converting a word to its base form. 
    Lemmatization considers context and converts the word to its meaningful base form
    The advantage of this is, we get to reduce the total number of unique words in the dictionary.
    Used Wordnet Lemmatizer

    Stemming -
    Removes last few characters, often leading to incorrect meaning
    Not used
    ‘Caring’ -> Lemmatization -> ‘Care’
    ‘Caring’ -> Stemming -> ‘Car’

    2) Feature vector generation
    Used - CountVectorizer 
    document-word matrix - will be denser with lesser columns. It convert text documents to a matrix of token counts

    3) Number of Topics
    Used Grid Search to find optimal Number of Topics
    Grid search do all combinations based on search params

    4) LDA model Implementation
    Latent Dirichlet Allocation (LDA) model is used.
    The probabilistic topic model estimated by it consists of two matrix.
    The first table describes the probability of selecting a particular part when sampling a particular topic. The second table describes the chance of selecting a particular topic when sampling a particular document

    5) Finding representative keywords
    Used lda_model.components_ which contains keywords and weights in a matrix of each topic
    Shown Top 10 Keyword

    6) Evaluation measure using Log Likelihood Score
    Model with higher log-likelihood is considered to be good. It considers the context and semantic associations between words