import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen

import random
import numpy as np
from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm.notebook.tqdm as tqdm
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import pickle
import re
import pandas as pd
import numpy as np
import sys
#!{sys.executable} -m pip install plotly
#import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# load all metadata
#hau nire diskan daukat SAD karpetan 2020-2021

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def cleantext(text):
    text= text.lower()
    text= re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text= re.sub(r"http\S+", "",text)
    text= re.sub(r"http", "",text)
    text= re.sub(r"usairways", "",text)
    
    punctuations= '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text= text.replace(p, '')
        
    text= [word.lower() for word in text.split() if word.lower() not in sw]
    
    text= [lemmatizer.lemmatize(word) for word in text]
    
    text = " ".join(text)
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F1E0-\U0001F1FF"  
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text= emoji_pattern.sub(r'',text)
    
    return text

   
#dfMergedfMeta= pd.read_csv('Cell_Phones/SamsungAppleXiaomiReviews_prueba.csv')
ml_dataset= pd.read_csv('./TweetsTrainDev.csv')
print(ml_dataset)
ml_dataset = ml_dataset[['airline_sentiment','text','airline']]

ml_dataset = ml_dataset[ml_dataset['airline'] == 'US Airways']

ml_dataset.drop('airline', axis=1, inplace=True)

print(ml_dataset)

dfUSAirwaysPositives=ml_dataset[ml_dataset['airline_sentiment'] == 'positive']
dfUSAirwaysNegatives=ml_dataset[ml_dataset['airline_sentiment'] == 'negative']


#GENERAR DOCUMENTOS
documentsPositives=dfUSAirwaysPositives[dfUSAirwaysPositives['text'].notna()]['text'].tolist()
documentsNegatives=dfUSAirwaysNegatives[dfUSAirwaysNegatives['text'].notna()]['text'].tolist()
#print(documentsPositives[:10])
#print(documentsNegatives[:10])

no_topics = 7 #@param {type:"integer"}
no_top_words = 30 #@param {type:"integer"}
no_top_documents = 5 #@param {type:"integer"}

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(''.join([' ' +feature_names[i] + ' ' + str(round(topic[i], 5)) #y esto también
                for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        docProbArray=np.argsort(W[:,topic_idx])
        #print(docProbArray)
        howMany=len(docProbArray)
        print("How Many")
        print(howMany)
        for doc_index in top_doc_indices:
            print(documents[doc_index])
            

#NEGATIVES
print("NEGATIVES-----------------------------------------------")
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
#print(documentsNegatives)
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documentsNegatives)
tf_feature_names = tf_vectorizer.get_feature_names_out()

# Run LDA
lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=100, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_W = lda_model.transform(tf) #lista de los documentos
inercia=lda_model.perplexity(lda_W,sub_sampling=False)
print("INERCIA NEGATIVOS ------ N.Topics=" + str(no_topics) + " --> " + str(inercia))
lda_H=lda_model.components_ /lda_model.components_.sum(axis=1)[:, np.newaxis]  #lista de las palabras
print("LDA Topics")
display_topics(lda_H, lda_W, tf_feature_names, documentsNegatives, no_top_words, no_top_documents)

#POSITIVES
print("POSITIVES-----------------------------------------------")
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documentsPositives)
tf_feature_names = tf_vectorizer.get_feature_names_out()

# Run LDA
lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=100, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_W = lda_model.transform(tf)

lda_H=lda_model.components_ /lda_model.components_.sum(axis=1)[:, np.newaxis] 
print("LDA Topics")
display_topics(lda_H, lda_W, tf_feature_names, documentsPositives, no_top_words, no_top_documents)