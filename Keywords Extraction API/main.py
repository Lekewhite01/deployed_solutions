import nltk
from nltk import tokenize

import re
import json
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

import os
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import string

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
from flask import Flask, request
from flask import jsonify

# Gensim
import gensim, spacy, warnings
import gensim.corpora as corpora
from gensim.models import TfidfModel

nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = list(set(stopwords.words('english')))
stop_words.extend(['lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right',
'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do',
'done', 'try', 'many','from', 'subject', 're', 'edu','some', 'nice', 'thank',
'think', 'see', 'rather', 'easy', 'easily', 'lot', 'line', 'even', 'also', 'may', 'take', 'come'])
exclude = set(string.punctuation)

stemmer = SnowballStemmer(language='english')
lemma = WordNetLemmatizer()

app = Flask(__name__)

@app.route('/post_data', methods=['POST'])
def index():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'text/plain'):
        data = str(request.data)
        text_sentences = tokenize.sent_tokenize(data)
        doc_complete = text_sentences
        doc_clean = [clean_data(doc).split() for doc in doc_complete]
        # Create Dictionary
        id2word = corpora.Dictionary(doc_clean)
        # Create Corpus
        texts = doc_clean
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        # corpus_test = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
        ldamodel = lda_train(corpus,id2word)
        df_topic_sents_keywords = pd.DataFrame()
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    df_topic_sents_keywords = df_topic_sents_keywords.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        df_topic_sents_keywords.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output

        contents = pd.Series(texts)
        df_topic_sents_keywords = pd.concat([df_topic_sents_keywords, contents], axis=1)
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        #

        # Display setting to show more characters in column
        pd.options.display.max_colwidth = 100

        sent_topics_sorteddf_mallet = pd.DataFrame()
        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                     grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                    axis=0)

        # Reset Index
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

        # Format
        sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
        main_list =  sent_topics_sorteddf_mallet['Keywords'].tolist()

        return jsonify(tags=main_list)

    else:
        return 'Content-Type not supported!'

def clean_data(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def lda_train(corpus,id2word):
    """funtion create an instance of lda model using gensim and training occur
    in: corpus,id2word and other hyperparameters
    out: trained lda model
    """
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=5,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    return lda_model

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
