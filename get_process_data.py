import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import pandas as pd

def data_process(shuff_data):
    stop_words = create_stop_word_list()
    shuff_clean_data = pd.DataFrame(shuff_data['doc'].apply(lambda x: get_cleaned_data(x,stop_words)),columns=['doc'])
    return shuff_clean_data

def create_stop_word_list():
    stp_words_eng = stopwords.words('English')
    extra_stop_words =['said','one','would','get','also','could','make','mr','want','told','take','may','n','like']
    en = spacy.load('en_core_web_sm')
    spacy_stopwords = list(en.Defaults.stop_words)
    for i in extra_stop_words:
        stp_words_eng.append(i)
    for i in spacy_stopwords:
        stp_words_eng.append(i)
    return stp_words_eng


def get_cleaned_data(line_data,stop_words):
#     print(line_data)
    no_spcl_char = del_hypr_link(line_data)
    tokenize_no_stopword_data = tokenize_del_stopword(no_spcl_char,stop_words)
    stem_line_data = get_stemming(tokenize_no_stopword_data)
#     print(stem_line_data)
    return stem_line_data

def del_hypr_link(val):
    """
    Removing hyperlinks and special characters
    """
#     print("del_hypr_link \n",val)
    new_val = re.sub(r'http\S+', '', val)
    
    #Special charactercls include  '', ~,!, #, % and integers
    text = re.sub("[^A-Za-z]+", " ", new_val)
    return text

def tokenize_del_stopword(val,stop_words):
    """
    Tokenize the word and remove the english stopword from each sentence.
    """
#     print("tokenize_del_stopword \n",val)
    tokens = nltk.word_tokenize(val)
#     print(tokens)
    tokens = [w for w in tokens if not w.lower() in stop_words]
    text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
#     print(text)
    return text

def get_stemming(val):
#     print('get_stemming',val)
#     s = PorterStemmer()
#     new_text = [s.stem(v) for v in val.split(' ')]
#     print('new_text',new_text)
#     final_text = ' '.join(new_text)
#     print('final_text',final_text)

    stemmer = SnowballStemmer("english")
    new_text = [stemmer.stem(v) for v in val.split(' ')]
    final_text = ' '.join(new_text)
    return final_text