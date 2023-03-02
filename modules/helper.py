import numpy as np

def doCombiStep(step, field, axis) -> np.ndarray:
    if(step == 'max'):
        return np.max(field, axis=axis)
    elif (step == 'sum'):
        return np.sum(field, axis=axis)

#flatten an 3D np array
def flatten(X, pos = -1):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    if pos == -1:
        pos = X.shape[1]-1
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, pos, :]
    return(flattened_X)

# Scale 3D array. X = 3D array, scalar = scale object from sklearn. Output = scaled 3D array.
def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

# Symbolize a 3D array. X = 3D array, scalar = SAX symbolizer object. Output = symbolic 3D string array.
def symbolize(X, scaler):
    X_s = scaler.transform(X)
    #X_s = X.astype('U13') 
    #for i in range(X.shape[0]):
        #X_s[i, :, :][:,0] = scaler.transform(np.array([X[i, :, :][:,0]]))
        #X_t.append(' '.join(X_s[i, :, :][:,0]))
    return X_s

# translate the a string [a,e] between 
def trans(val, vocab) -> float:
    for i in range(len(vocab)):
        if val == vocab[i]:
            halfSize = (len(vocab)-1)/2
            return (i - halfSize) / halfSize
    return -2

def getMapValues(size):
    vMap = []
    for i in range(size):
        halfSize = (size-1)/2
        vMap.append((i - halfSize) / halfSize)
    return vMap

def symbolizeTrans(X, scaler, sinfo, bins = 5):
    vocab = sinfo._check_params(bins)
    X_s = scaler.transform(X.tolist())
    X = np.zeros(X_s.shape)
    #X_s = X.astype(str) 
    for i in range(X.shape[0]):
        X = X.astype(float)
        
        #z1 = scaler.transform(np.array([X[i, :, :][:,0]]))
        
        for j in range(X.shape[1]):
            X[i][j] = trans(X_s[i][j], vocab)
    return X

def symbolizeTrans2(X, scaler, bins = 5):
    vocab = scaler._check_params(bins)
    for i in range(X.shape[0]):
        #X = X.astype('U13')
        X_s = X.astype(str) 
        z1 = scaler.transform(np.array([X[i, :, :][:,0]]))
        X_s[i, :, :][:,0] = z1
        for j in range(X.shape[1]):
            X[i][j][0] = trans(X_s[i][j][0], vocab)
    return X

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        np.save(f, obj)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return np.load(f, allow_pickle=True)
    
def truncate(n):
    return int(n * 1000) / 1000


def modelFidelity(modelPrediction, interpretationPrediction):
    summer = 0
    for i in range(len(modelPrediction)):
        if modelPrediction[i] == interpretationPrediction[i]:
            summer += 1
    return summer / len(modelPrediction)

def fillOutDicWithNNOutSmall(outData):
    inputDict = dict()
    inputDict['Val Accuracy'] = outData[0][0]
    inputDict['Val Precision'] = outData[0][1]
    inputDict['Val Recall'] = outData[0][2]
    inputDict['Val F1'] = outData[0][3]
    inputDict['Test Accuracy'] = outData[1][0]
    inputDict['Test Precision'] = outData[1][1]
    inputDict['Test Recall'] = outData[1][2]
    inputDict['Test F1'] = outData[1][3]
    inputDict['Train Predictions'] = outData[2][0]
    inputDict['Val Predictions'] = outData[2][1]
    inputDict['Test Predictions'] = outData[3]

    return inputDict