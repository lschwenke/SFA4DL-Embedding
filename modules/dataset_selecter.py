from tslearn.datasets import UCR_UEA_datasets
from sklearn.utils import shuffle
import numpy as np
from random import randint
from time import sleep

#select one dataset
def datasetSelector(dataset, seed_Value, number, takeName = True, use_cache=True):
    if dataset == 'ucr':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, number = doUCR(seed_Value, number, takeName = takeName, use_cache=use_cache)
    else:
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, number = []

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test = np.array(y_test)
    y_test = y_test.astype(float)
    X_test = np.array(X_test)
    X_test = X_test.astype(float)
    X_train = np.array(X_train)
    X_train = X_train.astype(float)   
    y_testy = np.array(y_testy)
    y_trainy = np.array(y_trainy)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, number




def doUCR(seed_value, number, takeName = True, retry=0, use_cache=True):
    try:
        datasets = UCR_UEA_datasets(use_cache=use_cache)
        dataset_list = datasets.list_univariate_datasets()
        if takeName:
            print(str(number))
            datasetName = number
            number = dataset_list.index(datasetName)
        else:
            print(str(number))
            datasetName = dataset_list[number]
        
        X_train, y_trainy, X_test, y_testy = datasets.load_dataset(datasetName)
        #X_train, y_trainy, X_test, y_testy = datasets.load_dataset('SyntheticControl')
        
        setY = list(set(y_testy))
        setY.sort()
        print(setY)

        num_of_classes = len(set(y_testy))
        seqSize = len(X_train[0])

        X_train, y_trainy = shuffle(X_train, y_trainy, random_state = seed_value)

        y_train = []
        print(num_of_classes)
        for y in y_trainy:
            y_train_puffer = np.zeros(num_of_classes)
            y_train_puffer[setY.index(y)] = 1
            y_train.append(y_train_puffer)

        y_trainy = np.argmax(y_train,axis=1) +1 
            
        y_test = []
        for y in y_testy:
            y_puffer = np.zeros(num_of_classes)
            y_puffer[setY.index(y)] = 1
            y_test.append(y_puffer)
            
        y_testy = np.argmax(y_test,axis=1) +1 
    
    except Exception as e:
        print(e)
        if retry < 5:
            sleep(randint(10,30))

            if retry == 4:
                return doUCR(seed_value, number, takeName = takeName, retry=retry+1, use_cache=False)
            else:
                return doUCR(seed_value, number, takeName = takeName, retry=retry+1) 

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, datasetName, num_of_classes, number

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks