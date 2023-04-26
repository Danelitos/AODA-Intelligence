#COSAS A CAMBIAR:
#variable target
#target_map
#cuando es binario, cambiar f1_score a None, sino 'weighted'
#SI VIENE CON EL TARGET O NO, comentar lineas necesarias

#llamadas: $ python plantillaTest.py -f SantanderTraHalfHalf.csv -m maxDepth45_mss1_msl1.sav

import csv
import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB,GaussianNB,MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


model=""
p="./"
preprocesado=0
vectorizar="bow"
undersample=0
oversample=0

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def cleantext(text):
    text= text.lower()
    text= re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text= re.sub(r"http\S+", "",text)
    text= re.sub(r"http", "",text)
    
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'p:m:f:h',['path=','model=','testFile=','h','preproceso=','vectorizar=','train=','undersample=','oversample='])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-p','--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('--preproceso'):
            preprocesado = int(arg)
        elif opt in ('--vectorizar'):
            vectorizar = arg.lower()
        elif opt in ('--undersample'):
            undersample = int(arg)
        elif opt in ('--oversample'):
            oversample = int(arg)
        elif opt in ('--train'):
            ficheroTrain = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n --train (fichero del train para vectorizar)\n --vectorizar bow o tfidf\n--preproceso 0 o 1\n --oversampling 0 o 1\n --undersampling 0 o 1\n' )
            exit(1)

    
    if p == './':
        model=p+str(m)
        iFile = p+ str(f)
    else:
        model=p+"/"+str(m)
        iFile = p+"/" + str(f)
        

    # astype('unicode') does not work as expected
    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)
    
    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    y_test=pd.DataFrame()
    testX = pd.read_csv(iFile) #el dataframe del testX
    ml_datasetTrain = pd.read_csv(ficheroTrain)
    
    testX = testX[['airline_sentiment','text']]
    ml_datasetTrain = ml_datasetTrain[['airline_sentiment', 'text']]
    #del testX['Class']
    #features=['Area','Perimeter','Compactness','kernelLength','KernelWidth','AsymmetryCoeff','KernelGrooveLength']
    target='airline_sentiment'
    # Lo de arriba son las columnas del DataSet que vamos a tener en cuenta 

    #para quitar el target si lo tuviera
    testY=testX[target]
    
    print(testY)
    target_map = {'positive': 0, 'neutral': 1,'negative':2}
    testY = testY.map(str).map(target_map)
    
    
    ml_datasetTrain['__target__'] = ml_datasetTrain[target].map(str).map(target_map)  # Mapeo del dataset de testeo
    del testX[target] #si viene con el target
   
    
    ml_datasetTrain = ml_datasetTrain[~ml_datasetTrain['__target__'].isnull()]
    
    train, dev = train_test_split(ml_datasetTrain,test_size=0.2,random_state=50,stratify=ml_datasetTrain[['__target__']])
    trainY=train[target]
    trainY = trainY.map(str).map(target_map)
    del train[target]
    #FEATURES-----------------------------------------------

    #Los FEATURES, son las columnas que vamos a tener en cuenta para el modelo, separados por tipo

    categorical_features = testX.select_dtypes(include=['category']).columns.tolist()
    numerical_features = testX.select_dtypes(include=['number']).columns.tolist()
    print("Numerical features: " + str(numerical_features))
    print("Categorical features: " + str(categorical_features))
    text_features = testX.select_dtypes(include=['object']).columns.tolist()

    #para cada FEATURE, realizarle la conversion a UNICODE

    for feature in categorical_features:
        testX[feature] = testX[feature].apply(coerce_to_unicode)

    for feature in text_features:
        testX[feature] = testX[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if testX[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(testX[feature].dtype, 'base') and testX[feature].dtype.base == np.dtype('M8[ns]')):
            testX[feature] = datetime_to_epoch(testX[feature]) #de dataiku
            #PASAR DE TIMESTAMP A EPOCH EN PANDAS (epoch es el tiempo que ha pasado desde 1970(clock del ordenador))
            testX[feature] = pd.date_range(testX[feature])
        else:
            testX[feature] = testX[feature].astype('double')
    
    
    #PREPROCESADO, si el modelo lo necesita----------------------------------------------------
    
    if preprocesado==1:
        
        #TRATAR MISSING VALUES----------------------------------

        #lista para eliminar filas que tienen missing values
        drop_rows_when_missing = []

        #lista para imputar filas que tienen missing values(mean por ejemplo)
        #impute_when_missing = [{'feature': 'Area', 'impute_with': 'MEAN'}, {'feature': 'Perimeter', 'impute_with': 'MEAN'},{'feature': 'Compactness', 'impute_with': 'MEAN'},{'feature': 'kernelLength', 'impute_with': 'MEAN'},{'feature': 'KernelWidth', 'impute_with': 'MEAN'},{'feature': 'AsymmetryCoeff', 'impute_with': 'MEAN'},{'feature': 'KernelGrooveLength', 'impute_with': 'MEAN'}]

        
        impute_when_missing=[]
        imputacion='MEAN'
        #añadir los features a la lista impute_when_missing para imputarlos
        for feature in numerical_features:
            valor={'feature':feature,'impute_with':imputacion}
            impute_when_missing.append(valor)
            
        # Features for which we drop rows with missing values"
        # quita las instancias de los missing values de la lista drop_rows_when_missing
        for feature in drop_rows_when_missing:
            testX = testX[testX[feature].notnull()]
            print('Dropped missing records in %s' % feature)

        # IMPUTA LOS VALORES DE la lista impute_when_missing con el valor de la elegido 
        for feature in impute_when_missing:
            if feature['impute_with'] == 'MEAN':
                v = testX[feature['feature']].mean()
            elif feature['impute_with'] == 'MEDIAN':
                v = testX[feature['feature']].median()
            elif feature['impute_with'] == 'CREATE_CATEGORY':
                v = 'NULL_CATEGORY'
            elif feature['impute_with'] == 'MODE':
                v = testX[feature['feature']].value_counts().index[0]
            elif feature['impute_with'] == 'CONSTANT':
                v = feature['value']
            testX[feature['feature']] = testX[feature['feature']].fillna(v)
            print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


        #para hacer el escalado de los atributos
        #rescale_features = {'Area': 'AVGSTD', 'Perimeter': 'AVGSTD', 'Compactness': 'AVGSTD', 'kernelLength': 'AVGSTD','KernelWidth': 'AVGSTD','AsymmetryCoeff': 'AVGSTD','KernelGrooveLength': 'AVGSTD'}
        
        rescale_features={}
        rescalado='AVGSTD'
        for feature in numerical_features: #obtenemos los features numericos
            rescale_features[feature]=rescalado
        print(rescale_features)
        
        # rescala los features de la lista, dependiendo del metodo que haya puesto(MINMAX, AVGSTD)
        for (feature_name, rescale_method) in rescale_features.items():
            if rescale_method == 'MINMAX':
                _min = testX[feature_name].min()
                _max = testX[feature_name].max()
                scale = _max - _min
                shift = _min
            else:
                shift = testX[feature_name].mean()
                scale = testX[feature_name].std()
            if scale == 0.:
                del testX[feature_name]
    
                print('Feature %s was dropped because it has no variance' % feature_name)
            else:
                print('Rescaled %s' % feature_name)
                testX[feature_name] = (testX[feature_name] - shift).astype(np.float64) / scale

        for feature in text_features:
            for feature in text_features:
                train[feature]=train[feature].apply(lambda x: cleantext(x))
                testX[feature]=testX[feature].apply(lambda x: cleantext(x))
            
        
        print("TEXTO PREPROCESADO -----------------------")
        print(testX.head(5))
    
    print(oversample)
    if undersample==1:
        print("HACIENDO UNDERSAMPLE...")
        undersample=RandomUnderSampler(sampling_strategy="not minority",random_state=50)
        train,trainY = undersample.fit_resample(train,trainY)
    
    elif oversample==1:
        print("HACIENDO OVERSAMPLE...")
        undersample=RandomUnderSampler(sampling_strategy="not majority",random_state=50)
        train,trainY= undersample.fit_resample(train,trainY)

    print(vectorizar)
    if (vectorizar=="tfidf"):
        #features de TEXTO, realizamos tf-idf o BOW
        tfidf_vectorizer = TfidfVectorizer() 
        tfidf_train_vectors = tfidf_vectorizer.fit_transform(train['text'])
        trainX=tfidf_train_vectors.toarray()
        print(tfidf_train_vectors.shape)
        tfidf_test_vectors = tfidf_vectorizer.transform(testX['text'])
        testX=tfidf_test_vectors.toarray()
    
    elif (vectorizar=="bow"):    
        #REALIZAR EL vectorizer con el BOW
        count_vectorizer=CountVectorizer()
        tfidf_train_vectors = count_vectorizer.fit_transform(train['text'])
        trainX=tfidf_train_vectors.toarray()
        print(tfidf_train_vectors.shape)
        tfidf_test_vectors = count_vectorizer.transform(testX['text'])
        testX=tfidf_test_vectors.toarray()
    
    # tfidf_vectorizer = TfidfVectorizer() 
    # print(testX)
    # tfidf_test_vectors = tfidf_vectorizer.transform(['@AmericanAir how about some rampers at gate b40 dfw?   Waiting to be marshaled in'])
    # testX=tfidf_test_vectors.toarray()
    # print(tfidf_test_vectors.shape)
    
    #print(testX.head(5))
    clf = pickle.load(open(model, 'rb'))
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    y_test['preds'] = predictions
    #predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    #results_test = testX.join(predictions, how='left')
    print(predictions)
    
    ficheroPredicciones=open("predicciones.csv",'w', newline='')
    writer=csv.writer(ficheroPredicciones)
    
    for prediction in predictions:
        writer.writerow([prediction])
    
    #print(results_test)
    #f1Score=f1_score(testY, predictions, average='weighted') #'weighted' o None  #SI VIENE CON EL TARGET
    #print("F1-Score: " + str(f1Score)) #SI VIENE CON EL TARGET