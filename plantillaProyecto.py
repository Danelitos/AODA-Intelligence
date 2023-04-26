#COSAS A CAMBIAR:
#variable target
#target_map
#cuando es binario, cambiar f1_score a None, sino 'weighted'

#llamadas knn: $ python plantillaModelos.py -a knn -k 1 -K 7 -d 1 -D 2 -w 1 -W 2 --missingValues 0 --preproceso 0 --undersample 1 -f iris.csv -o resumen.csv

#llamadas decision tree: $ python plantillaModelos.py -a decisionTree --md1 3 --md2 9 --msl1 1 --msl2 2 --mss1 1 --mss2 2 --missingValues 0 --preproceso 0 --undersample 0 -f iris.csv -o resumen.csv

# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import pickle
import re
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #algoritmo de arbol de decision
from sklearn.naive_bayes import CategoricalNB,GaussianNB,MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import KBinsDiscretizer
import csv
import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#from fs.time import datetime_to_epoch

#DATOS POR DEFECTO
p='./'
#f="trainHalfHalf.csv"  #nombre del archivo
f="datasetForTheExam_SubGrupo1.csv"
oFile="resumen.csv"
algoritmo="naivebayes" #por defecto
tipo="continuos" #para NAIVE BAYES
vectorizar="tfidf"

#VARIABLES knn
#valor de k(vecinos)
kMin=9
kMax=9
#valor de p(distancia)
pMin=1
pMax=2
#valor de w(pesos) 1=uniform , 2=distance
wMin=1
wMax=2 


#VARIABLES DecisionTree
maxDepth1=3
maxDepth2=15

mss1=1
mss2=2

msl1=1
msl2=2

criterio="gini" #criterio gini o entropy

#MAS VARIABLES
preprocesado=0
missingValues=0
undersample=0
oversample=0

# Press the green button in the gutter to run the script.

#AL EJECUTARLO, LE PODEMOS PASAR PARAMETROS:
#La k,p...

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

def trainAndResults(clf,modelName): #imprimir y guardar los resultados con el modelo, para despues guardar el mejor
    clf.class_weight = "balanced"

    # Entrenar el modelo    
    clf.fit(trainX, trainY)

    
    # Build up our result dataset
    # The model is now trained, we can apply it to our test set:

    predictions = clf.predict(testX) #predicciones realizadas por el modelo
    probas = clf.predict_proba(testX) #probabilidad

    # predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    # cols = [
    #     u'probability_of_value_%s' % label
    #     for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    # ]
    # probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

    # # Build scored dataset
    # results_test = testX.join(predictions, how='left')
    # results_test = results_test.join(probabilities, how='left')
    # results_test = results_test.join(test['__target__'], how='left')
    # results_test = results_test.rename(columns= {'__target__': 'TARGET'})

    # #print(results_test)
    
    # i=0
    # for real,pred in zip(testY,predictions):
    #     #print(real,pred)
    #     i+=1
    #     if i>5:
    #         break
    
    #mostramos todos los resultados obtenidos

    print(f"Resultados para: " + modelName[:-4])
    #print(testY)
    f1Score=f1_score(testY, predictions, average='weighted') #el f-score del modelo(None para binaria, 'weighted' para multiclass)
    print("F1-Score: " + str(f1Score))
    
    resumen=classification_report(testY,predictions, output_dict=True) #resultado en modo dict.
    #print(resumen) 
    
    print(confusion_matrix(testY, predictions)) #binario [1,0]
    print('-------------------------------')
    #añadir los resultados al fichero csv
    
    writer = csv.writer(ficheroResumen)
    writer.writerow([modelName[:-4]])
    for key,datos in resumen.items():
        #print(datos)
        if key!="accuracy":
            writer.writerow([key,datos.get("precision"),datos.get("recall"),datos.get("f1-score")])
        else:
            writer.writerow([key,datos])
    
    writer.writerow([]) #linea en blanco en el csv

    #crear el par de valores para despues obtener el que tenga mejor f-score         
    if type(f1Score)==np.ndarray: #cuando la clase es binaria y se pone average=None, devuelve un array con el valor de cada clase
        print("ES BINARIA")
        f1Score=f1Score[0]
    par=(f1Score,modelName,clf) #se guardan con el formato de tripleta (f-score,nombre,modelo), para despues obtener el mayor f-score, y guardarlo con el nombre y el modelo
    modelos.append(par) #añadirlo a la lista de modelos



#---------------------------------------------MAIN ----------------------------------------------

if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:a:k:K:d:D:w:W:p:f:h',['output=','algoritmo=','missingValues=','preproceso=','k=','K=','d=','D=','w=','W=','md1=','md2=','msl1=','msl2=','mss1=','mss2=','path=','iFile','h','undersample=','oversample=','tipo=','vectorizar='])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt in ('-a','--algoritmo'):
            algoritmo = arg.lower() 
        elif opt in ('--missingValues'):
            missingValues=int(arg)  
        elif opt in ('--preproceso'):
            preprocesado=int(arg)
            
        #parametros knn
        elif opt == '-k':
            kMin = int(arg)
        elif opt in ('-K','--K'):
            kMax = int(arg)
        elif opt ==  '-d':
            pMin = int(arg)
        elif opt ==  '-D':
            pMax = int(arg)
        elif opt ==  '-w':
            wMin = int(arg)
        elif opt ==  '-W':
            wMax = int(arg)
            
        #parametros decisionTree
        elif opt ==  '--md1':
            maxDepth1 = int(arg)
        elif opt ==  '--md2':
            maxDepth2 = int(arg)
        elif opt ==  '--msl1':
            msl1 = int(arg)
        elif opt ==  '--msl2':
            msl2 = int(arg)
        elif opt ==  '--mss1':
            mss1 = int(arg)
        elif opt ==  '--mss2':
            mss2 = int(arg)
            
        elif opt == '--undersample':
            undersample = int(arg)
        elif opt ==  '--oversample':
            oversample = int(arg)
        elif opt ==  '--tipo':
            tipo = arg.lower()
        elif opt ==  '--vectorizar':
            vectorizar = arg.lower()
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print('\n --algoritmo knn o decisiontree \n \n ALGORITMO KNN variables: \n -k -K min y max \n -d -D distancia min y max \n -w -W 1=uniform, 2=distance \n ALGORITMO DECISIONTREE variables: \n --md1 maxDepth minimo \n --md1 maxDepth maximo \n --mss1 min_samples_split minimo \n --mss2 min_samples_split maximo \n --msl1 min_samples_leaf minimo \n --msl2 min_samples_split maximo \n --missingValues : 0 no(defecto), 1 si, para tratar missing values \n --preproceso : 0 no(defecto), 1 si \n --undersample : 0 no(defecto), 1 si \n --oversample : 0 no(defecto), 1 si \n \n -o outputFile \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)

    if p == './':
        iFile=p+str(f)
    else:
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

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    

    #LAS COLUMNAS DEL DATASET QUE TENEMOS EN CUENTA (FEATURES + COLUMNA DEL MODELO(target))
    
    #----------------CONFIGURACION PARA EL ENTRENAMIENTO----------------------------------------

    ml_dataset = ml_dataset[['airline_sentiment','text']]
    #ml_dataset = ml_dataset[ml_dataset.columns.tolist()]
    #IMPRIMIR LOS PRIMEROS n FILAS DEL DATASET
    print(ml_dataset.head(5))
    #del ml_dataset['']
    
    #para eliminar una columna
    #del ml_dataset['Largo de petalo']
    #print(ml_dataset)
    #features=['Largo de sepalo','Ancho de sepalo','Largo de petalo','Ancho de petalo']
    
    target='airline_sentiment' #columna del target
    # Lo de arriba son las columnas del DataSet que vamos a tener en cuenta 

    #FEATURES-----------------------------------------------

    #Los FEATURES, son las columnas que vamos a tener en cuenta para el modelo, separados por tipo

    categorical_features = ml_dataset.drop(target,axis=1).select_dtypes(include=['category']).columns.tolist()
    numerical_features = ml_dataset.drop(target,axis=1).select_dtypes(include=['number']).columns.tolist()
    print("Numerical features: " + str(numerical_features))
    print("Categorical features: " + str(categorical_features))
    text_features = ml_dataset.drop(target,axis=1).select_dtypes(include=['object']).columns.tolist()
    print("Text features: " + str(text_features))
    #stamps='2012-10-08 18:15:05'
    #print(pd.Timestamp("1970-01-01"))
    #print((stamps - pd.Timestamp("1970-01-01"))/pd.Timedelta("1s"))

    #para cada FEATURE, realizarle la conversion a UNICODE

    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature]) #de dataiku
            #PASAR DE TIMESTAMP A EPOCH EN PANDAS (epoch es el tiempo que ha pasado desde 1970(clock del ordenador))
            ml_dataset[feature] = pd.date_range(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')

    
    #PASAR DE CATEGORIAL A NUMERICO
    #TARGET ES LA COLUMNA que queremos PREDECIR

    #PASAR DE CATEGORIAL A NUMERICO los valores del TARGET
    target_map = {'positive': 0, 'neutral': 1,'negative':2}
    ml_dataset['__target__'] = ml_dataset[target].map(str).map(target_map) #hacer el mapeo de texto a numero del target
    print("target" + str(ml_dataset['__target__']))
    #eliminar la columna target que tiene el texto
    del ml_dataset[target]
    

    # Remove rows for which the target is unknown.
    #quitar las columnas que no estan en el target_map(nulas o desconocidas)
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print("FILE:")
    print(f)
    print("ROWS: ")
    print(ml_dataset.head(5))
    print(ml_dataset.count())
    #CREAR LOS DATOS DEL TRAIN Y DEL TEST
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=50,stratify=ml_dataset[['__target__']])
    print(train.head(5))
    #CONTADOR DE CADA VALOR DEL TARGET value_counts() hace el recuento de cada clase
    print("VALORES DEL TRAIN:")
    print(train['__target__'].value_counts())
    print("VALORES DEL TEST: ")
    print(test['__target__'].value_counts())

    #INFO train.shape[0] (numero de filas) ; train.shape[1] (numero de columnas)
    
    if missingValues==1:
        
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
            train = train[train[feature].notnull()]
            test = test[test[feature].notnull()]
            print('Dropped missing records in %s' % feature)

        # IMPUTA LOS VALORES DE la lista impute_when_missing con el valor de la lista 
        for feature in impute_when_missing:
            if feature['impute_with'] == 'MEAN':
                v = train[feature['feature']].mean()
            elif feature['impute_with'] == 'MEDIAN':
                v = train[feature['feature']].median()
            elif feature['impute_with'] == 'CREATE_CATEGORY':
                v = 'NULL_CATEGORY'
            elif feature['impute_with'] == 'MODE':
                v = train[feature['feature']].value_counts().index[0]
            elif feature['impute_with'] == 'CONSTANT':
                v = feature['value']
            train[feature['feature']] = train[feature['feature']].fillna(v)
            test[feature['feature']] = test[feature['feature']].fillna(v)
            print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


    if preprocesado==1:
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
                _min = train[feature_name].min()
                _max = train[feature_name].max()
                scale = _max - _min
                shift = _min
            else:
                shift = train[feature_name].mean()
                scale = train[feature_name].std()
            if scale == 0.:
                del train[feature_name]
                del test[feature_name]
                print('Feature %s was dropped because it has no variance' % feature_name)
            else:
                print('Rescaled %s' % feature_name)
                train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
                test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

        #hacer preprocesado de TEXTO
        for feature in text_features:
            train[feature]=train[feature].apply(lambda x: cleantext(x))
            test[feature]=test[feature].apply(lambda x: cleantext(x))
        
        print("TEXTO PREPROCESADO -----------------------")
        print(train.head(5))
            
    #SEPARAR el TRAIN Y TEST en FEATURES y TARGET

    #quitar del train y test el target (solo quedan los features)

    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']
    print(trainX.head(5))
    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    #añadir el target
    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])


    # hace un underSample/overSample
    #undersampling --> cuando tenemos muchos datos
    #oversampling --> cuando tenemos pocos datos (crea datos duplicados)
    #undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    
    if undersample==1:
        print("HACIENDO UNDERSAMPLE...")
        undersample=RandomUnderSampler(sampling_strategy="not minority",random_state=50)
        trainX,trainY = undersample.fit_resample(trainX,trainY)
        testX,testY = undersample.fit_resample(testX, testY)
    
    elif oversample==1:
        print("HACIENDO OVERSAMPLE...")
        undersample=RandomUnderSampler(sampling_strategy="not majority",random_state=50)
        trainX,trainY = undersample.fit_resample(trainX,trainY)
        testX,testY = undersample.fit_resample(testX, testY)

    modelos=[] #array para guardar (f-score,modelo)
    ficheroResumen=open(oFile,'w', newline='')
    writer=csv.writer(ficheroResumen)
    writer.writerow(["ModelName","Precision","Recall","F-Score"])
    
    #------------------------------------------------------------------VECTORIZAR EL TEXTO----------------------------------------------------------------------
    if (vectorizar=="tfidf"):
        #features de TEXTO, realizamos tf-idf o BOW
        tfidf_vectorizer = TfidfVectorizer() 
        print(trainX)
        tfidf_train_vectors = tfidf_vectorizer.fit_transform(trainX['text'])
        trainX=tfidf_train_vectors.toarray()
        print(tfidf_train_vectors.shape)
        tfidf_test_vectors = tfidf_vectorizer.transform(testX['text'])
        testX=tfidf_test_vectors.toarray()
    
    elif (vectorizar=="bow"):    
        #REALIZAR EL vectorizer con el BOW
        count_vectorizer=CountVectorizer()
        tfidf_train_vectors = count_vectorizer.fit_transform(trainX['text'])
        trainX=tfidf_train_vectors.toarray()
        print(tfidf_train_vectors.shape)
        tfidf_test_vectors = count_vectorizer.transform(testX['text'])
        testX=tfidf_test_vectors.toarray()
    
    #------------------------------------------------------------------ALGORITMOS -------------------------------------------------
    
    if algoritmo=="knn":
        
        for k in range(kMin,kMax+1,2):
            for p in range(pMin,pMax+1,1):
                for i in range(wMin,wMax+1,1):
                    if i==1:
                        w='uniform'
                    else:
                        w='distance'

                    # Crear el modelo con el algoritmo especificado
                    clf = KNeighborsClassifier(n_neighbors=k,
                                        weights=w,
                                        algorithm='auto',
                                        leaf_size=30,
                                        p=2)

                    # algoritmo de decision tree
                    modelName = "k" + str(k) + "_p" + str(p) + "_" + w + ".sav"
                    
                    #hacer la llamada a la funcion para entrenar y obtener los resultados
                    trainAndResults(clf,modelName)
    
    elif algoritmo=="decisiontree":
        
        for md in range(maxDepth1,maxDepth2+1,3):
            for mss in range(mss1,mss2+1,1):
                for msl in range(msl1,msl2+1,1):
                    clf = DecisionTreeClassifier(
                                random_state = 1337,
                                criterion = criterio,    #gini o entropy
                                splitter = 'best',       #best o random
                                max_depth = md,           #numero 3,6,9 ....
                                min_samples_split = mss, #1 o 2
                                min_samples_leaf = msl   #1 o 2
                        )
                    
                    modelName = "maxDepth" + str(md) + "_mss" + str(mss) + "_msl" + str(msl) + ".sav"
                    
                    #hacer la llamada a la funcion para entrenar y obtener los resultados
                    trainAndResults(clf,modelName)
                    
    elif algoritmo=="naivebayes":
        for i in range(1,5):
            if tipo=="categoriales":
                clf = CategoricalNB()
                modelName = "naiveCategoriales" + ".sav"
            
            elif tipo=="continuos":
                clf = GaussianNB()
                modelName = "naiveContinuos" + ".sav"
                
            elif tipo=="multinomial":
                clf = MultinomialNB(alpha=i)
                modelName = "naiveMultinomial" + ".sav"
            elif tipo=="bernoulli":
                clf = BernoulliNB(alpha=i)
                modelName = "naiveBernoulli" + ".sav"    
                #clf.fit(tfidf_train_vectors, trainY)
                
            trainAndResults(clf,modelName)
        
    #aqui ya tenemos los modelos con su f-score
    #cojemos el modelo con maximo f-score
    
    mejorModelo = max(modelos, key=lambda x: x[0])
    
    #estructura mejorModelo=(f-score,nombreModelo,clf)
      
    #tenemos el mejor modelo
    print("El mejor modelo: " + str(mejorModelo))
    
    #guardamos el modelo con pickle
    saved_model = pickle.dump(mejorModelo[2], open(mejorModelo[1],'wb'))
    

print("ENTRENAMIENTO FINALIZADO, BUENA SUERTE CON EL TEST!")
