import get_data
import get_process_data
import math
import perform_cluster

import pandas as pd
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt 


def perform_split_data(processed_data):
    l = len(processed_data)
    l_train = math.floor(l*.80)
    train_data = processed_data[:l_train]
    test_data = processed_data[l_train+1:]
    return train_data,test_data

def word_cloud(text,wc_file_name='wordcloud.jpeg'):
    # Create stopword list
    stopword_list = set(STOPWORDS) 

    # Create WordCloud 
    word_cloud = WordCloud(width = 800, height = 500, 
                           background_color ='white', 
                           stopwords = stopword_list, 
                           min_font_size = 14).generate(text) 

    # Set wordcloud figure size
    plt.figure(figsize = (8, 6)) 
    
    # Set title for word cloud
    plt.title('wc_title')
    
    # Show image
    plt.imshow(word_cloud) 

    # Remove Axis
    plt.axis("off")  

    # save word cloud
    plt.savefig(wc_file_name,bbox_inches='tight')

    # show plot
    plt.show()

shuff_data = get_data.get_various_data()
processed_data = get_process_data.data_process(shuff_data)
split_train_data,split_test_data = perform_split_data(processed_data)
kmeans_cluster_df = perform_cluster.generate_model(split_train_data)
top_words = perform_cluster.get_top_words_cluster(kmeans_cluster_df)

for i in kmeans_cluster_df.cluster_label.unique():
    new_df=kmeans_cluster_df[kmeans_cluster_df.cluster_label==i]
#     print(new_df)
    text="".join(new_df.doc.tolist())
#     print(text)
    word_cloud(text)


# print("\n\n top words",top_words)




