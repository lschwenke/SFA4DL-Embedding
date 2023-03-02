from sacred import Experiment
import numpy as np
import seml
import os
import random
import warnings

from modules import helper
from modules import dataset_selecter as ds
from modules import modelCreator

from datetime import datetime


import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    #init before the experiment!
    @ex.capture(prefix="init")
    def baseInit(self, nrFolds: int, patience: int, seed_value: int, symbolCount: int):
        self.seed_value = seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
        tf.random.set_seed(seed_value)
        np.random.RandomState(seed_value)

        np.random.seed(seed_value)

        context.set_global_seed(seed_value)
        ops.get_default_graph().seed = seed_value

        #pip install tensorflow-determinism needed
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed_value)

        #save some variables for later
        self.symbolCount = symbolCount

        self.valuesA = helper.getMapValues(symbolCount)
        self.kf = StratifiedKFold(nrFolds, shuffle=True, random_state=seed_value)
        self.earlystop = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=patience, verbose=0, mode='auto')
        self.fold = 0
        self.nrFolds = nrFolds
        self.seed_value = seed_value        

        #init gpu
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices: 
            tf.config.experimental.set_memory_growth(gpu_instance, True)


    # Load the dataset
    @ex.capture(prefix="data")
    def init_dataset(self, dataset: str, number: int, takename: bool):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        self.number = number
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_trainy, self.y_testy, self.seqSize, self.dataName, self.num_of_classes, self.number = ds.datasetSelector(dataset, self.seed_value, number, takeName=takename)


    #all inits
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.baseInit()
        self.init_dataset()

    #methods to save the results into the fullResults dict

    def fillOutDicWithNNOutFull(self, abstractionString, configString, fullResults, inputDict):
        if abstractionString not in fullResults.keys():
            fullResults[abstractionString] = dict()
        if configString not in fullResults[abstractionString].keys():
            fullResults[abstractionString][configString] = []
        fullResults[abstractionString][configString].append(inputDict)



    def fillOutDicWithNNOut(self, abstractionString, configString, fullResults, outData):
        inputDict = helper.fillOutDicWithNNOutSmall(outData)
        self.fillOutDicWithNNOutFull(abstractionString, configString, fullResults, inputDict)

    def printTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)


    # one experiment run with a certain config set. MOST OF THE IMPORTANT STUFF IS DONE HERE!!!!!!!
    @ex.capture(prefix="model")
    def trainExperiment(self, numEpochs: int, batchSize: int, useEmbed: bool, calcOri: bool, calcSFA: bool, doSymbolify: bool, useSaves: bool, useShapeSaves: bool, multiVariant: bool, skipDebugSaves: bool, 
        dropeOutRate: float, dmodel: int, dff: int, header: int, numOfAttentionLayers: int, limit: int, ncoef: list, strategy):

        print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
        print('Dataname:')
        print(self.dataName)
        self.printTime()
        warnings.filterwarnings('ignore')   

        fullResults = dict()


        #if you know which configs already finished you can activate this
        #wname = modelCreator.getWeightName(self.dataName, self.number, self.symbolCount, numOfAttentionLayers, "results", header, dmodel=dmodel, dff=dff, doDetails=True, learning = False, results = True, posthoc=ncoef[0], resultsPath = 'otherResults')
        if False and not os.path.isfile(wname + '.pkl'):
            fullResults["Error"] = "dataset " + self.dataName + " not included: " + str(self.seqSize) + "; name: " + wname
            print('Not included ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " not included: " + str(self.seqSize) + "; name: " + wname)

            return "dataset " + self.dataName + " not included " + str(self.seqSize)  + "; name: " + wname #fullResults
        
        #don't recalculate already finished experiments
        wname = modelCreator.getWeightName(self.dataName, self.number, self.symbolCount, numOfAttentionLayers, "results", header, dmodel=dmodel, dff=dff, doDetails=True, learning = False, results = True, posthoc=ncoef[0], resultsPath = 'results')
        if os.path.isfile(wname + '.pkl'):
            fullResults["Error"] = "dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname
            print('Already Done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname)
        
            return "dataset " + self.dataName + "already done: " + str(self.seqSize)  + "; name: " + wname #fullResults

        #limit the lenght of the data
        toLong = False
        if self.seqSize > limit:
            if calcSFA and ncoef[0] != 0:
                toLong = True
            else:
                fullResults["Error"] = "dataset " + self.dataName + " to big: " + str(self.seqSize)
                print('TO LONGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print("dataset " + self.dataName + " to big: " + str(self.seqSize))
                return "dataset " + self.dataName + " to big: " + str(self.seqSize) #fullResults

        # k fold train loop
        for train, test in self.kf.split(self.X_train, self.y_trainy):
            self.fold+=1            

            #preprocess data
            x_train1 = self.X_train[train]
            x_val = self.X_train[test]
            y_train1 = self.y_train[train]
            y_trainy2 = self.y_trainy[train]
            y_val = self.y_train[test]
            
            x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy = modelCreator.preprocessData(x_train1, x_val, self.X_test, y_train1, y_val, self.y_test, y_trainy2, self.y_testy, self.fold, self.symbolCount, self.dataName, useEmbed = useEmbed, useSaves = useShapeSaves, doSymbolify = doSymbolify, multiVariant=multiVariant, strategy=strategy)

            # calc original model
            abstractionString = "Original"
            if(calcOri and not toLong):
                
                outOri = modelCreator.doAbstractedTraining(X_train_ori, X_val_ori, X_test_ori, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves, useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                abstractionType=abstractionString, rate=dropeOutRate)
                self.fillOutDicWithNNOut(abstractionString, "results", fullResults, outOri)
            self.printTime()

            # calc SAX or SFA model
            if (not calcSFA) or (ncoef[0] == 0):
                abstractionString = "SAX"
                outSax = modelCreator.doAbstractedTraining(x_train1, x_val, x_test, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves, useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                    abstractionType=abstractionString, rate=dropeOutRate)
                self.printTime()
                self.fillOutDicWithNNOut(abstractionString, "results", fullResults, outSax)   
            else:
                limitS = self.seqSize/ncoef[1]
                if ncoef[0] > limitS:
                    ncoefI = int(limitS)
                else:
                    ncoefI = ncoef[0]


                x_train1 = self.X_train[train]
                x_val = self.X_train[test]
                y_train1 = self.y_train[train]
                y_trainy2 = self.y_trainy[train]
                y_val = self.y_train[test]

                x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy = modelCreator.preprocessData(x_train1, x_val, self.X_test, y_train1, y_val, self.y_test, y_trainy2, self.y_testy, self.fold, self.symbolCount, self.dataName, useEmbed = useEmbed, useSaves = useShapeSaves, doSymbolify = doSymbolify, multiVariant=multiVariant, doSFA=True, ncoef=ncoefI, strategy=strategy)

                abstractionString = "SFA" + str(ncoef[1])
                outSax = modelCreator.doAbstractedTraining(x_train1, x_val, x_test, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves,useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                    abstractionType=abstractionString, rate=dropeOutRate, ncoef=ncoef[0])
                self.printTime()
                self.fillOutDicWithNNOut(abstractionString, "results", fullResults, outSax)
                fullResults['ncoef']= ncoef
            predictions = np.argmax(y_train1,axis=1) +1

            print("finished fold: " + str(self.fold))
            self.printTime()         
        print("Done done")
        saveName = modelCreator.getWeightName(self.dataName, self.number, self.symbolCount, numOfAttentionLayers, "results", header, dmodel=dmodel, dff=dff,doDetails=True, learning = False, results = True, posthoc=ncoef[0])
        print(saveName)
        helper.save_obj(fullResults, str(saveName))
        

        self.printTime()

        return saveName


  

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.trainExperiment()