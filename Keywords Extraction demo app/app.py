import nltk
from nltk import tokenize

import re
import matplotlib.colors as mcolors
import pickle
# import builtins
import streamlit as st
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
from gensim.models.wrappers import LdaMallet
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import string

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
import streamlit as st


# Gensim
import gensim, spacy, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import TfidfModel

st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore",category=DeprecationWarning)

stop_words = list(set(stopwords.words('english')))
stop_words.extend(['lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right',
'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do',
'done', 'try', 'many','from', 'subject', 're', 'edu','some', 'nice', 'thank',
'think', 'see', 'rather', 'easy', 'easily', 'lot', 'line', 'even', 'also', 'may', 'take', 'come'])
exclude = set(string.punctuation)
#lemmatization and stemming
# nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
nltk.download('stopwords')
stemmer = SnowballStemmer(language='english')
lemma = WordNetLemmatizer()

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


def display_app_header(main_txt, sub_txt, is_sidebar=False):
    """
    Code Credit: https://github.com/soft-nougat/dqw-ivves
    function to display major headers at user interface
    :param main_txt: the major text to be displayed
    :param sub_txt: the minor text to be displayed
    :param is_sidebar: check if its side panel or major panel
    :return:
    """

    html_temp = f"""
    <h2 style = "text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "text_align:center;"> {sub_txt} </p>
    </div>
    """

    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    else:
        st.markdown(html_temp, unsafe_allow_html=True)

def divider():
    """
    Sub-routine to create a divider for webpage contents
    """

    st.markdown("""---""")

@st.cache
def preprocess(doc):
    return clean_data(doc)

@st.cache(allow_output_mutation=True)
def lda_mod(corpus,id2word):
    return lda_train(corpus,id2word)

def main():
    st.write("""
    # Keyword Extraction with Insights7
    In this application, we'll see how the most dominant topic keywords from a body of text is extracted.
    Let's start by entering a sample text
    """)

    input_help_text = """
    Enter any text of your choice
    """

    final_message = """
    The keywords from the input text were successfully extracted
    """
    text = st.text_area(label='INPUT TEXT',placeholder="Enter Sample Text")

    with st.sidebar:
        # st.image(Image.open("../data/image_data/start.png"))
        # st.markdown("**Step 1**")
        st.markdown("**Processing**")
        start_process = st.checkbox(
            label="Start",
            help="Starts the Insights7 Extraction"
        )

    if start_process:
        # Fancy Header
        # Slik Wrangler default header
        display_app_header(
            main_txt='Insights7 Demo Web Application',
            sub_txt='Input -> Process -> Extract Topic Keywords -> Plot wordcloud'
        )
        divider()

        if text is not None:
            st.info('Text loaded', icon="ℹ️")
#
            if st.sidebar.checkbox("Process Text"):
                with st.spinner('Wait for it...'):
                    data = preprocess(text)
                    text_sentences = tokenize.sent_tokenize(text)
                    doc_complete = text_sentences
                    doc_clean = [clean_data(doc).split() for doc in doc_complete]
                    # Create Dictionary
                    id2word = corpora.Dictionary(doc_clean)
                    # Create Corpus
                    texts = doc_clean
                    # Term Document Frequency
                    corpus = [id2word.doc2bow(text) for text in texts]
                    # corpus_test = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
                    ldamodel = lda_mod(corpus,id2word)
                    df_topic_sents_keywords = pd.DataFrame()

                    # Get main topic in each document
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

                    # Format
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

                    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

                    cloud = WordCloud(stopwords=stop_words,
                                      background_color='white',
                                      width=2500,
                                      height=1800,
                                      max_words=10,
                                      colormap='tab10',
                                      color_func=lambda *args, **kwargs: cols[i],
                                      prefer_horizontal=1.0)

                    topics = ldamodel.show_topics(formatted=False)

                    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

                    for i, ax in enumerate(axes.flatten()):
                        fig.add_subplot(ax)
                        topic_words = dict(topics[i][1])
                        cloud.generate_from_frequencies(topic_words, max_font_size=300)
                        plt.gca().imshow(cloud)
                        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
                        plt.gca().axis('off')


                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.axis('off')
                    plt.margins(x=0, y=0)
                    plt.tight_layout()
                    plot = plt.show()

                    st.pyplot(plot)

                    # st.dataframe(sent_topics_sorteddf_mallet)

                    st.success('Topic keywords extracted successfully!!!')
                    # # st.snow()


if __name__ == '__main__':
    main()
