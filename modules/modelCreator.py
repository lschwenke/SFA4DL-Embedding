import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from modules import helper
from modules import transformer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pyts.approximation import SymbolicAggregateApproximation
from pyts.approximation import SymbolicFourierApproximation
import math
import os

#create the transformer model with given information
def createModel(splits, x_train, x_val, x_test, batchSize, seed_value, num_of_classes, numOfAttentionLayers, dmodel, header, dff, rate = 0.0, useEmbed=False):    
        x_trains = np.dsplit(x_train, splits)
        print(np.array(x_trains).shape)

        x_trainsBatch = np.dsplit(x_train[:batchSize], splits)

        x_tests = np.dsplit(x_test, splits)
        x_vals = np.dsplit(x_val, splits)
        maxLen = len(x_trains[0][0])
        print(maxLen)

        if(useEmbed):
            for i in range(splits):
                x_trains[i]= np.array([[" ".join([item[0] for item in x])] for x in x_trains[i]])
                x_trainsBatch[i]= np.array([[" ".join([item[0] for item in x])] for x in x_trainsBatch[i]])
                x_tests[i]= np.array([[" ".join([item[0] for item in x])] for x in x_tests[i]])
                x_vals[i]= np.array([[" ".join([item[0] for item in x])] for x in x_vals[i]])

        print(np.array(x_trains).shape)
        flattenArray = []
        inputShapes = []
        encClasses = []
        for i in range(len(x_trains)):
            mask = Input(1)
            x_part = np.array(x_trains[i])
            print(np.array(x_part).shape)
        
            seq_len1 = x_part.shape[1]

            if(useEmbed):
                left_input1 = Input(shape=(1,), dtype=tf.string)
            else:
                sens1 = x_part.shape[2]
                input_shape1 = (seq_len1, sens1)
                left_input1 = Input(input_shape1)

            inputShapes.append(left_input1)

            if(useEmbed):
                encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
                    max_tokens=50, output_sequence_length=maxLen)
                encoder.adapt(x_part)

                encoded = encoder(left_input1)
                print(encoder.get_vocabulary())
                input_vocab_size = len(np.array(encoder.get_vocabulary()))
            else:
                encoded = left_input1
                input_vocab_size = 0
                
            #create transformer encoder layer 
            if(useEmbed):
                #rate=0.38
                encClass1 = transformer.Encoder(numOfAttentionLayers, dmodel, header, dff, maxLen, rate=rate, input_vocab_size = input_vocab_size + 2, maxLen = maxLen, doEmbedding=useEmbed, seed_value=seed_value)
            else:
                encClass1 = transformer.Encoder(numOfAttentionLayers, dmodel, header, dff, 5000, rate=rate, input_vocab_size = input_vocab_size + 2, maxLen = maxLen, seed_value=seed_value)
                
            encClasses.append(encClass1)

            maskLayer = tf.keras.layers.Masking(mask_value=-2)
            encInput = maskLayer(encoded)
            enc1, attention, fullAttention = encClass1(encInput)
            print(enc1.shape)
            flatten1 = Flatten()(enc1)
            flattenArray.append(flatten1)
        

        # Merge nets
        if splits == 1:
            merged = flattenArray[0]
        else:
            merged = concatenate(flattenArray)

        output = Dense(num_of_classes, activation = "sigmoid")(merged)
        
        # Create combined model
        wdcnnt_multi = Model(inputs=inputShapes,outputs=(output))
        print(wdcnnt_multi.summary())
        
        print(wdcnnt_multi.count_params())
        
        tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed_value)

        learning_rate = transformer.CustomSchedule(32)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.99, 
                                     epsilon=1e-9)
        
        wdcnnt_multi.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['accuracy'], run_eagerly=False)
        
        print('done')
        
        return wdcnnt_multi, inputShapes, x_trains, x_tests, x_vals

# building saving name for model weights
def getWeightName(name, fold, symbols, layers, abstractionType, header,  dmodel=None, dff=None, learning = True, resultsPath = 'results', results=False, usedp=False, doHeaders=True, doDetails=False, posthoc=None):
    #baseName = "./saves/weights-" + str(data_path_train.split('/')[-1].split('.')[0]) + '-size' + str(seqSize) + '-threshold' + maxString + '-input' + abstractionString + '-fold' + str(fold) + '-bins' + str(n_bins)

    if usedp:
        if results:
            baseName = "./"+resultsPath+"/results-" +name +' -f: '+str(fold)+' -s: '+str(symbols)+ ' -l: '+str(layers)+' -a: '+abstractionType
        else: 
            baseName = "./saves/weights-" +name +' -f: '+str(fold)+' -s: '+str(symbols)+ ' -l: '+str(layers)+' -a: '+abstractionType
    else:
        if results:
            baseName = "./"+resultsPath+"/results-" +name +' -f '+str(fold)+' -s '+str(symbols)+ ' -l '+str(layers)+' -a '+abstractionType
        else: 
            baseName = "./saves/weights-" +name +' -f '+str(fold)+' -s '+str(symbols)+ ' -l '+str(layers)+' -a '+abstractionType
    
    if doHeaders:
        baseName = baseName + ' -h ' + str(header)
    if doDetails:
        baseName = baseName + ' -dm ' + str(dmodel)
        baseName = baseName + ' -df ' + str(dff)
    if posthoc:
        baseName = baseName + ' -p' + str(posthoc)
    if learning:
        return baseName + '-learning.tf'
    else:
        return baseName + '.tf'

# do training for the given model def
def doAbstractedTraining(trainD, valD, testD, y_train1, y_val, y_testy, BATCH, seed_value, num_of_classes, dataName, fold, symbolCount, num_epochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=False, useEmbed = False, 
    earlystop = None, useSaves=False, abstractionType=None, rate=0.0, ncoef=0):

    
    newTrain = trainD
    newVal = valD
    newTest = testD

    print('newTrain during abstract training:')
    print(newTrain.shape)
            
    n_model2, inputs2, x_trains2, x_tests2, x_vals2 = createModel(1, newTrain, newVal, newTest , BATCH, seed_value, num_of_classes, numOfAttentionLayers, dmodel, header, dff, rate=rate)
    
    print(np.array(x_trains2).shape)
    print(np.array(x_vals2).shape)
    
    if (os.path.isfile(getWeightName(dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, dmodel=dmodel, dff=dff, doDetails=True, learning=False, posthoc=ncoef) + '.index') and useSaves):
        print('found weights to load! Won\'t train model!')
        n_model2.load_weights(getWeightName(dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, dmodel=dmodel, dff=dff, doDetails=True, learning=False, posthoc=ncoef))
    else:
        print('No weights found! Start training model!')
        n_model2.fit(x_trains2, y_train1, validation_data = (x_vals2, y_val) , epochs = num_epochs, batch_size = BATCH, verbose=1, callbacks =[earlystop], shuffle = True)
        print('Model fitted!!!')
        n_model2.save_weights(getWeightName(dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, dmodel=dmodel, dff=dff, doDetails=True, learning=False, posthoc=ncoef), overwrite=True)
        
    #print([lay.name for lay in n_model.layers])
    print('Starting earlyPredictor creation')
    if(useEmbed):
        earlyPredictor2 = tf.keras.Model(n_model2.inputs, n_model2.layers[3].output)
    else:
        earlyPredictor2 = tf.keras.Model(n_model2.inputs, n_model2.layers[2].output)

    # Predictions on the validation set
    print('starting val prediction')
    predictions2 = n_model2.predict(x_vals2)
    
    #print(len(attentionQ))
    #print(attentionQ[1])
    print('############################')
    predictions2 = np.argmax(predictions2,axis=1)+1

    # Measure this fold's accuracy on validation set compared to actual labels
    y_compare = np.argmax(y_val, axis=1)+1
    val_score2 = metrics.accuracy_score(y_compare, predictions2)
    val_pre = metrics.precision_score(y_compare, predictions2, average='macro')
    val_rec = metrics.recall_score(y_compare, predictions2, average='macro')
    val_f1= metrics.f1_score(y_compare, predictions2, average='macro')

    print(f"validation fold score with input {abstractionType}(accuracy): {val_score2}")

    # Predictions on the test set
    limit = 500
    test_predictions_loop2 = []
    for bor in range(int(math.ceil(len(x_tests2[0])/limit))):
        test_predictions_loop2.extend(n_model2.predict([x_tests2[0][bor*limit:(bor+1)*limit]]))

    attentionQ2 = None
    if (not skipDebugSaves):
        attentionQ0 = []
        attentionQ1 = []
        attentionQ2 = []

        for bor in range(int(math.ceil(len(x_trains2[0])/limit))):
            attOut = earlyPredictor2.predict([x_trains2[0][bor*limit:(bor+1)*limit]])
            attentionQ0.extend(attOut[0]) 
            attentionQ1.extend(attOut[1])

            if len(attentionQ2) == 0:
                attentionQ2 = attOut[2]
            else:
                for k in range(len(attentionQ2)):
                    attentionQ2[k] = np.append(attentionQ2[k], attOut[2][k], 0)


        attentionQ2 = [attentionQ0, attentionQ1, attentionQ2]

    test_predictions_loop2 = np.argmax(test_predictions_loop2, axis=1)+1

    # Measure this fold's accuracy on test set compared to actual labels
    test_score2 = metrics.accuracy_score(y_testy, test_predictions_loop2)
    test_pre = metrics.precision_score(y_testy, test_predictions_loop2, average='macro')
    test_rec = metrics.recall_score(y_testy, test_predictions_loop2, average='macro')
    test_f1= metrics.f1_score(y_testy, test_predictions_loop2, average='macro')

    print(f"test fold score with input {abstractionType}-(accuracy): {test_score2}")

    train_predictions= []
    for bor in range(int(math.ceil(len(x_trains2[0])/limit))):
        train_predictions.extend(n_model2.predict([x_trains2[0][bor*limit:(bor+1)*limit]]))
    train_predictions = np.argmax(train_predictions, axis=1)+1

    if skipDebugSaves:
        return [val_score2, val_pre, val_rec, val_f1], [test_score2, test_pre, test_rec, test_f1], [train_predictions, predictions2], test_predictions_loop2, None, None, None, None, None, None, None, None, None, y_val, y_train1
    else:
        return [val_score2, val_pre, val_rec, val_f1], [test_score2, test_pre, test_rec, test_f1], [train_predictions, predictions2], test_predictions_loop2, n_model2, x_trains2, x_tests2, x_vals2, attentionQ2, earlyPredictor2, newTrain, newVal, newTest, y_val, y_train1





def preprocessData(x_train1, x_val, X_test, y_train1, y_val, y_test, y_trainy, y_testy, binNr, symbolsCount, dataName, useEmbed = False, useSaves = False, doSymbolify = True, multiVariant=False, doSFA=False, ncoef=125, strategy='uniform'):    
    
    x_test = X_test.copy()
    
    if(useEmbed):
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount) + '+embedding'
    else:
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount)
    fileExists = os.path.isfile(processedDataName +'.pkl')

    if(fileExists and useSaves):
        print('found file! Start loading file!')
        res = helper.load_obj(processedDataName)


        for index, v in np.ndenumerate(res):
            print(index)
            res = v
        res.keys()

        x_train1 = res['X_train']
        #x_train1 = res['X_val']
        x_test = res['X_test']
        x_val = res['X_val']
        X_train_ori = res['X_train_ori']
        X_test_ori = res['X_test_ori']
        y_trainy = res['y_trainy']
        y_train1 = res['y_train']
        y_test = res['y_test']
        y_testy = res['y_testy']
        y_val = res['y_val']
        X_val_ori = res['X_val_ori']
        print(x_test.shape)
        print(x_train1.shape)
        print(y_test.shape)
        print(y_train1.shape)
        print('SHAPES loaded')
        
    else:
        print('SHAPES:')
        print(x_test.shape)
        print(x_train1.shape)
        print(x_val.shape)
        print(y_test.shape)
        print(y_train1.shape)

        x_train1 = x_train1.squeeze()
        x_val = x_val.squeeze()
        x_test = x_test.squeeze()
        
        trainShape = x_train1.shape
        valShape = x_val.shape
        testShape = x_test.shape
        
        if multiVariant:
            X_test_ori = x_test.copy()
            X_val_ori = x_val.copy()
            X_train_ori = x_train1.copy()
            for i in range(trainShape[-1]):
                x_train2 = x_train1[:,:,i]
                x_val2 = x_val[:,:,i]
                x_test2 = x_test[:,:,i]
                print('####')
                print(x_train2.shape)

                trainShape2 = x_train2.shape
                valShape2 = x_val2.shape
                testShape2 = x_test2.shape
        
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train2 = scaler.transform(x_train2.reshape(-1, 1)).reshape(trainShape2)##
                x_val2 = scaler.transform(x_val2.reshape(-1, 1)).reshape(valShape2)
                x_test2 = scaler.transform(x_test2.reshape(-1, 1)).reshape(testShape2)

                if doSFA:
                    sax = SymbolicFourierApproximation(n_coefs=ncoef,n_bins=symbolsCount, strategy=strategy)
                else:
                    sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy=strategy)
                sinfo = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy=strategy)
                sax.fit(x_train2)

                if(useEmbed):
                    x_train2 = helper.symbolize(x_train2, sax)
                    x_val2 = helper.symbolize(x_val2, sax)
                    x_test2 = helper.symbolize(x_test2, sax)
                else:
                    x_train2 = helper.symbolizeTrans(x_train2, sax, sinfo=sinfo, bins = symbolsCount)
                    x_val2 = helper.symbolizeTrans(x_val2, sax, sinfo=sinfo, bins = symbolsCount)
                    x_test2 = helper.symbolizeTrans(x_test2, sax, sinfo=sinfo, bins = symbolsCount)
                print(x_train2.shape)
                #x_train1 = np.expand_dims(x_train1, axis=2)

                x_train1[:,:,i] = x_train2      
                x_val[:,:,i] = x_val2
                x_test[:,:,i] = x_test2
                
            #x_train1 = x_train1.reshape(trainShape[0],-1,1)
            #x_val = x_val.reshape(valShape[0],-1,1)
            #x_test = x_test.reshape(testShape[0],-1,1)
            print(x_train1.shape)
            

        else:    
            if(doSymbolify):
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train1 = scaler.transform(x_train1.reshape(-1, 1)).reshape(trainShape)
                x_val = scaler.transform(x_val.reshape(-1, 1)).reshape(valShape)
                x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(testShape)

                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()


                if doSFA:
                    sax = SymbolicFourierApproximation(n_coefs=ncoef,n_bins=symbolsCount, strategy=strategy)
                else:
                    sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy=strategy)       
                sinfo = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy=strategy)
         
                sax.fit(x_train1)

                if(useEmbed):
                    x_train1 = helper.symbolize(x_train1, sax)
                    x_val = helper.symbolize(x_val, sax)
                    x_test = helper.symbolize(x_test, sax)
                else:
                    x_train1 = helper.symbolizeTrans(x_train1, sax, sinfo=sinfo, bins = symbolsCount)
                    x_val = helper.symbolizeTrans(x_val, sax, sinfo=sinfo, bins = symbolsCount)
                    x_test = helper.symbolizeTrans(x_test, sax, sinfo=sinfo, bins = symbolsCount)
            else:
                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()


            x_train1 = np.expand_dims(x_train1, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)   
            X_test_ori = np.expand_dims(X_test_ori, axis=2)   
            X_train_ori = np.expand_dims(X_train_ori, axis=2) 
            X_val_ori = np.expand_dims(X_val_ori, axis=2) 
            
            

        print('saves shapes:')
        print(x_test.shape)
        print(x_train1.shape)

        #save sax results to only calculate them once
        resultsSave = {
            'X_train':x_train1,
            'X_train_ori':X_train_ori,
            'X_test':x_test,
            'X_test_ori':X_test_ori,
            'X_val': x_val,
            'X_val_ori':X_val_ori,
            'y_trainy':y_trainy,
            'y_train':y_train1,
            'y_val': y_val,
            'y_test':y_test,
            'y_testy':y_testy
        }
        helper.save_obj(resultsSave, processedDataName)
    return x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy