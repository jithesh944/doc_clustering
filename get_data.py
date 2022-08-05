import pandas as pd
import random
import wikipedia
from sklearn.datasets import fetch_20newsgroups
import os


def get_various_data():
    """
    Function is to collect the various data from different websites like wikipedia, kaggle
    """
    path= os.getcwd()

    wiki_data = get_wiki_data()
    # print(wiki_data)
    built_sklearn_data = get_built_in_sklearn_data()
    mixed_txt_data = get_txt_data(path)
    sports_politics_data = get_kaggle_data(path)
    shuff_all_data = perform_merge_all_data(wiki_data,built_sklearn_data,mixed_txt_data,sports_politics_data)
    return shuff_all_data


def get_wiki_data():
    """
    Wikipedia library in oython enables us to get the data on the specfic given topics from wikipedia.
    Here, only 4 sentence on the each topics is selected.Each topic's data equivalent to a document in 
    information retrieval.
    """
    keys_topics =['COVID-19','World Health Organisation','food','Life','Gym','Antigen test','brexit',
                'boris johnson','shinzo abe','Justin Trudeau','Political science','Moralism','Political science',
                'Political system','Franklin D. Roosevelt','Head of government','cricket','football','Olympic Games',
                'Basketball','Swimming','Wimbledon Championships','Rugby union','Chess','Go (game)','Video game',
                'Esports','Formula One','Gymnastics','Diving (sport)','Asian Games','Commonwealth Games',
                'Festival of Empire','National Basketball Association']
    # keys_topics =['COVID-19']
    random.shuffle(keys_topics)
    wiki_lst=[]
    for article in keys_topics:
#     print("loading content: ",article)
        wiki_lst.append(wikipedia.summary(article,sentences=4))
    wiki_lst1 = [b for a in wiki_lst for b in a.split('.')]
    wiki_lst2 = list(filter(None,wiki_lst1))
    wiki_lst2 = pd.DataFrame(wiki_lst2,columns=['doc'])    
    return wiki_lst2


def get_built_in_sklearn_data():
    """
    Sklern has some bulit-in datasetswhith different topics such medicice, 
    sports(hockey,baseball),graphics, computers
    Here retrieveing only medical data
    """
    cats = ['sci.med']
    groups_train = fetch_20newsgroups(subset='train', categories=cats,shuffle=True,
                                    remove=('headers', 'footers', 'quotes'))
    doc_data_df = pd.DataFrame(groups_train.data,columns=['doc'])
    return doc_data_df



def get_txt_data(path):
    """
    The data is from some data from interent
    """
    df_txt = pd.read_table(os.path.join(path,'documentcw.txt'),header=None)
    df_txt = df_txt.rename(columns={0:'doc'})
    return df_txt


def get_kaggle_data(path):

    """
    getting the sports and poltics data from the file deowloaded from kaggle
    """
    df_csv = pd.read_csv(os.path.join(path,'poltics_alone.csv'),sep=',')
    return df_csv


def perform_merge_all_data(wiki_data,built_sklearn_data,mixed_txt_data,sports_politics_data):
    # print('wiki_data',wiki_data)
    # print('built_sklearn_data',built_sklearn_data)
    # print('mixed_txt_data',mixed_txt_data)
    # print('sports_politics_data',sports_politics_data)
    df = wiki_data['doc'].append(built_sklearn_data['doc'].append((mixed_txt_data['doc'].
                append(sports_politics_data['text'],ignore_index=True)),ignore_index=True),ignore_index=True)
    # df1 = pd.DataFrame(wiki_lst2.conc,df_csv.text,df.doc,columns='doc')print()
    df = pd.DataFrame(df,columns=['doc'])
    return df

if __name__ =='__main__':
    print(get_various_data())