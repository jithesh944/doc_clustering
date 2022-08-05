import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import get_process_data

def check_test_data(vectorizer,kmeans_model):
    test_data = ["players are ready to play their game by any chance",
    'celtic make late bid for bellamy newcastle striker craig bellamy is discussing a possible short-term loan move to celtic  bbc sport understands.  the welsh striker has rejected a move to birmingham after falling out with magpies manager graeme souness. the toon boss vowed bellamy would not play again after a bitter row over his exclusion for the game against arsenal. celtic are in no position to match birmingham s £6m offer but a stay until the end of the season could suit bellamy while he considers his future. according to bellamy s agent  the player dismissed a permanent move to birmingham. and it is unlikely that newcastle would allow the player to go on loan to another premiership club.  bellamy was fined two weeks  wages after a live tv interview in which he accused souness of lying  following a very public dispute about what position bellamy should play in the side. souness said:  he can t play for me ever again. he has been a disruptive influence from the minute i walked into this football club.  he can t go on television and accuse me of telling lies.  chairman freddy shepherd described bellamy s behaviour as  totally unacceptable and totally unprofessional.',
    'parties warned over  grey vote  political parties cannot afford to take older uk voters for granted in the coming election  says age concern.  a survey for the charity suggests 69% of over-55s say they always vote in a general election compared with just 17% of 18 to 24 year olds. charity boss gordon lishman said if a  decisive blow  was struck at the election it would be by older voters who could be relied on to turn out. a total of 3 028 adults aged 18 or over were interviewed for the study. mr lishman urged the next government to boost state pension.  he also called for measures to combat ageism and build effective public services to  support us all in an ageing society .  older people want to see manifesto commitments that will make a difference to their lives   mr lishman said.  political parties must wake up to the fact that unless they address the demands and concerns of older people they will not keep or attract their vote.  in the survey carried out by icm research  14% of people aged between 18 and 34 said they never voted in general elections. among the over-65s  70% said they would be certain to vote in an immediate election  compared with 39% of people under 55. age concern says the over-55s are  united around  key areas of policy they want the government to focus on. for 57%  pensions and the nhs were key issues  while the economy was important for a third  and tax was a crucial area for 25%.  the report was welcomed by conservative shadow pensions secretary david willetts.  the pensioners  voice must certainly be heard in the next election as they have never fitted into blair s cool britannia   he said.  labour s continued refusal to admit the true extent of the pensions crisis will be one of the monumental failures of this government.  he pointed to tory plans to increase the basic state pension to reduce means testing  strengthen company pensions and encourage savings. a liberal democrat spokesman said the party took the issues raised in the report very seriously. he highlighted the party s promises to raise the basic state pension  provide free long-term care for the elderly and replace council tax  seen as a particular problem for pensioners on fixed incomes. labour has said it wants to use savings reforms to incapacity benefit to improve the basic state pension and has set up a review of the council tax system.',
    'money bank in england is closing soon'
    ]
    td = pd.DataFrame(test_data,columns=['doc'])
    td= get_process_data.data_process(td)
    test_predict =list()

    Y = vectorizer.transform(td['doc'])
    prediction = kmeans_model.predict(Y)
    test_predict.append(prediction)
    print('The cluster value for document is---- ', test_predict)

def generate_model(df2_train):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df2_train['doc'])
    # print(len(X.todense()))

    kmeans_model = KMeans(n_clusters=3,)
    kmeans_model.fit(X)
    print( "Sum of squared distance for this model is : ",kmeans_model.inertia_)
    clusters_value = kmeans_model.labels_
    df2_train['cluster_label'] = clusters_value
    check_test_data(vectorizer,kmeans_model)
    return df2_train

def perform_grouping(clustered_data):

    sam = clustered_data.groupby(by =['cluster_label'])
    grp=dict()
    for i in range(sam.ngroups):
        name = "grp_" + str(i)
        grp[name] = sam.get_group(i).reset_index()
    #     print(name)
    return grp

def get_top_words_cluster(clustered_data):
    # Top 25 words each cluster
    grouped_data = perform_grouping(clustered_data)
    word_dict=dict()
    word_dict_final=dict()
    for k,v in grouped_data.items():
        # print(k)
        for l in range(len(v.doc)):
            line_split = v.doc[l].split()
#           print(line_split)
            temp_dict=dict()
            for i in range (0,len(line_split)-1):
                # print(line_split[i])
                if line_split[i] in word_dict:
                    word_dict[line_split[i]] += 1
                else:
                    word_dict[line_split[i]] = 1
            temp_dict.update(word_dict)  
        word_dict_final[k] = dict(sorted(temp_dict.items(),key=lambda x: x[1],reverse=True)[:25])

    return word_dict_final