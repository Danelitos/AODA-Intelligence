import io
import os.path
import re
import tarfile
import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import smart_open
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.coherencemodel import CoherenceModel



#variables de prepreceso
iFile = 'TweetsTrainDev.csv' 
empresaObj = 'US Airways'
procesarTexto = 1

ratingObj = 'negative'

#variables de train
num_topics_neg= 9 #2,5,8
num_topics_pos= 1 #3,8

chunksize = 2000
passes = 20
iterations = 400
eval_every = None 


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



    #funcion de topic modeling usando gensim

def GensimTopicModeling(docs,num_topics):
    print("Number of docs: "+str(len(docs)))
    #print(docs[0][:500])

    #####################################################################
            ############    PREPROCESO    ###############
    #####################################################################

    if procesarTexto ==1:
        for i in range (0,len(docs)):
            docs[i]=cleantext(docs[i])
            #print(docs[i])
                

    # Tokenize the documents.

    # Separar los documentos en tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = tokenizer.tokenize(docs[idx])  # Separar en palabras.
        #print(docs[idx])

    # Eliminar numeros individuales.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    # Eliminar palabras de un solo caracter.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    #Lematizar las palabras
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Compute bigrams.
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
        #print(docs[idx])

    #print(docs)
    #Eliminar tanto palabras frecuentes como infrecuentes
    dictionary = Dictionary(docs)
    #print(Dictionary)
    #dictionary.filter_extremes(no_below=20, no_above=0.5)


    #BOW
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    #tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    #corpus = tf_vectorizer.fit_transform(docs)
    #tf_feature_names = tf_vectorizer.get_feature_names_out()

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    #####################################################################
                ############    TRAIN    ###############
    #####################################################################


    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    top_topics = model.top_topics(corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    
    for i in range(0,len(top_topics)):
        pal=""
        topic=top_topics[i]
        print("TOPIC " + str(i))
        palabras=topic[0]
        for t in palabras:
            pal=pal+ str(t[1]) + ", "
            
        print(pal)
        print()
    #print(top_topics)
    cv=CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence='c_v')
    coherencia=cv.get_coherence()
    print("COHERENCIA: " + str(coherencia))


#Leer csv de entrada
df = pd.read_csv(iFile)




df_obj = df[df['airline']==empresaObj]
df_neg = df_obj[df_obj['airline_sentiment']=='negative']
df_neg = df_neg['text']
docsNeg = df_neg.to_numpy().tolist()



df_pos = df_obj[df_obj['airline_sentiment']=='positive']
df_pos = df_pos['text']
#print(df_obj.head(5))
docsPos = df_pos.to_numpy().tolist()


print("#####################################################################")
print("            ############    NEGATIVOS    ###############")
print("#####################################################################")
GensimTopicModeling(docsNeg,num_topics_neg)

print("#####################################################################")
print("            ############    POSITIVOS    ###############")
print("#####################################################################")
GensimTopicModeling(docsPos,num_topics_pos)


