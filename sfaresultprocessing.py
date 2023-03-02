import os
import random
import numpy as np
import math

from modules import helper
from tslearn.datasets import UCR_UEA_datasets


datasets = UCR_UEA_datasets(use_cache=True)
dataset_list = datasets.list_univariate_datasets()


#all variables to check
symbolsA = [5,6,7]
layersA = [2,4]
headersA = [8,16]
dmsA = [16,8]
ncoefsA = [256, 128,64,0]
folds=5
strategies=['uniform', 'quantile']

directory = './results/'

fresults = dict()


#for each dataset
for dNr in range(len(dataset_list)):
        fresults[dNr] = dict()
        dName = dataset_list[dNr]
        fresults[dNr]['name'] = dName
        
        # find best per type and overall and save it
        fresults[dNr]['best overall'] = dict()
        fresults[dNr]['best'] = dict()
        fresults[dNr]['best']['acc'] = -1 #0
        fresults[dNr]['best']['method'] = 'None'
        fresults[dNr]['best']['config'] = 'None'

        fresults[dNr]['best Ori'] = dict()
        fresults[dNr]['best Ori']['acc'] = -1 #0
        fresults[dNr]['best Ori']['config'] = 'None'

        fresults[dNr]['best SAX'] = dict()
        fresults[dNr]['best SAX']['acc'] = -1 #0
        fresults[dNr]['best SAX']['config'] = 'None'
        fresults[dNr]['best SAX']['diff'] = -1 #0

        fresults[dNr]['best SFA'] = dict()
        fresults[dNr]['best SFA']['acc'] = -1 #0
        fresults[dNr]['best SFA']['config'] = 'None'
        fresults[dNr]['best SFA']['diff'] = -1 #0

        for strategy in strategies:
                for symbols in symbolsA:
                        for layers in layersA:
                                for header in headersA:
                                        for dm in dmsA:
                                                dff = int(dm/2)
                                                for ncoef in ncoefsA:
                                                        f = os.path.join(directory, f'results-{dName} -f {dNr} -s {symbols} -l {layers} -a results -h {header} -dm {dm} -df {dff} -p{ncoef} -st{strategy[0:1]}.tf')
                                                        
                                                        config = f'-s {symbols} -l {layers} -a results -h {header} -dm {dm} -df {dff} -p{ncoef} -st{strategy[0:1]}'
                                                        fresults[dNr][config] = dict()
                                                        currentDict = fresults[dNr][config]
                                                        currentDict['best'] = dict()
                                                        currentDict['best']['acc'] = -1
                                                        currentDict['best']['method'] = 'None'
                                                        if os.path.isfile(f+'.pkl'):

                                                                loadedRes = helper.load_obj(f)
                                                                for index, v in np.ndenumerate(loadedRes):
                                                                        fres = v
                                                                
                                                                currentDict['Ori test acc'] = -1
                                                                currentDict['Ori test rec'] = -1
                                                                currentDict['Ori test prec'] = -1
                                                                currentDict['Ori test f1'] = -1
                                                                currentDict['SAX test acc'] = -1
                                                                currentDict['SAX test rec'] = -1
                                                                currentDict['SAX test prec'] = -1
                                                                currentDict['SAX test f1'] = -1
                                                                currentDict['SFA test acc'] = -1
                                                                currentDict['SFA test rec'] = -1
                                                                currentDict['SFA test prec'] = -1
                                                                currentDict['SFA test f1'] = -1
                                                                
                                                                if 'Original' in fres.keys():
                                                                        
                                                                        currentDict['Ori test acc'] = 0
                                                                        currentDict['Ori test rec'] = 0
                                                                        currentDict['Ori test prec'] = 0
                                                                        currentDict['Ori test f1'] = 0
                                                                        
                                                                        for e in fres['Original']['results']:
                                                                                currentDict['Ori test acc'] += e['Test Accuracy']
                                                                                currentDict['Ori test rec'] += e['Test Recall']
                                                                                currentDict['Ori test prec'] += e['Test Precision']
                                                                                currentDict['Ori test f1'] += e['Test F1']
                                                                                
                                                                        currentDict['Ori test acc'] /= len(fres['Original']['results'])
                                                                        currentDict['Ori test rec'] /= len(fres['Original']['results'])
                                                                        currentDict['Ori test prec'] /= len(fres['Original']['results'])
                                                                        currentDict['Ori test f1'] /= len(fres['Original']['results'])

                                                                        if currentDict['Ori test acc'] > currentDict['best']['acc']:
                                                                                currentDict['best']['method'] = 'Ori'
                                                                                currentDict['best']['acc'] = currentDict['Ori test acc']
                                                                        
                                                                        if currentDict['Ori test acc'] > fresults[dNr]['best']['acc']:
                                                                                fresults[dNr]['best']['config'] = config
                                                                                fresults[dNr]['best']['method'] = 'Ori'
                                                                                fresults[dNr]['best']['acc'] = currentDict['Ori test acc']

                                                                        if currentDict['Ori test acc'] > fresults[dNr]['best Ori']['acc']:
                                                                                fresults[dNr]['best Ori']['config'] = config
                                                                                fresults[dNr]['best Ori']['acc'] = currentDict['Ori test acc']

                                                                if 'ncoef' not in fres.keys():
                                                                        
                                                                        currentDict['SAX test acc'] = 0
                                                                        currentDict['SAX test rec'] = 0
                                                                        currentDict['SAX test prec'] = 0
                                                                        currentDict['SAX test f1'] = 0
                                                                        
                                                                        for e in fres['SAX']['results']:
                                                                                currentDict['SAX test acc'] += e['Test Accuracy']
                                                                                currentDict['SAX test rec'] += e['Test Recall']
                                                                                currentDict['SAX test prec'] += e['Test Precision']
                                                                                currentDict['SAX test f1'] += e['Test F1']
                                                                                
                                                                        currentDict['SAX test acc'] /= len(fres['SAX']['results'])
                                                                        currentDict['SAX test rec'] /= len(fres['SAX']['results'])
                                                                        currentDict['SAX test prec'] /= len(fres['SAX']['results'])
                                                                        currentDict['SAX test f1'] /= len(fres['SAX']['results'])

                                                                        if currentDict['SAX test acc'] > currentDict['best']['acc']:
                                                                                currentDict['best']['method'] = 'SAX'
                                                                                currentDict['best']['acc'] = currentDict['SAX test acc']

                                                                        if currentDict['SAX test acc'] > fresults[dNr]['best']['acc']:
                                                                                fresults[dNr]['best']['config'] = config
                                                                                fresults[dNr]['best']['method'] = 'SAX'
                                                                                fresults[dNr]['best']['acc'] = currentDict['SAX test acc']

                                                                        if currentDict['SAX test acc'] > fresults[dNr]['best SAX']['acc']:
                                                                                fresults[dNr]['best SAX']['config'] = config
                                                                                fresults[dNr]['best SAX']['acc'] = currentDict['SAX test acc']
                                                                else:
                                                                        sfaName = 'SFA' + str(fres['ncoef'][1])

                                                                        currentDict['SFA test acc'] = 0
                                                                        currentDict['SFA test rec'] = 0
                                                                        currentDict['SFA test prec'] = 0
                                                                        currentDict['SFA test f1'] = 0
                                                                        
                                                                        for e in fres[sfaName]['results']:
                                                                                currentDict['SFA test acc'] += e['Test Accuracy']
                                                                                currentDict['SFA test rec'] += e['Test Recall']
                                                                                currentDict['SFA test prec'] += e['Test Precision']
                                                                                currentDict['SFA test f1'] += e['Test F1']
                                                                                
                                                                        currentDict['SFA test acc'] /= len(fres[sfaName]['results'])
                                                                        currentDict['SFA test rec'] /= len(fres[sfaName]['results'])
                                                                        currentDict['SFA test prec'] /= len(fres[sfaName]['results'])
                                                                        currentDict['SFA test f1'] /= len(fres[sfaName]['results'])

                                                                        if currentDict['SFA test acc'] > currentDict['best']['acc']:
                                                                                currentDict['best']['method'] = 'SFA'
                                                                                currentDict['best']['acc'] = currentDict['SFA test acc']
                                                                        
                                                                        if currentDict['SFA test acc'] > fresults[dNr]['best']['acc']:
                                                                                fresults[dNr]['best']['config'] = config
                                                                                fresults[dNr]['best']['method'] = 'SFA'
                                                                                fresults[dNr]['best']['acc'] = currentDict['SFA test acc']

                                                                        if currentDict['SFA test acc'] > fresults[dNr]['best SFA']['acc']:
                                                                                fresults[dNr]['best SFA']['config'] = config
                                                                                fresults[dNr]['best SFA']['acc'] = currentDict['SFA test acc']
                                                                                
                                                        else:
                                                                print('failed to find: '+ str(f+'.pkl'))

        fresults[dNr]['best SAX']['diff'] = fresults[dNr]['best Ori']['acc'] - fresults[dNr]['best SAX']['acc']
        fresults[dNr]['best SFA']['diff'] = fresults[dNr]['best Ori']['acc'] - fresults[dNr]['best SFA']['acc']

helper.save_obj(fresults, "./sfaResults/finalOut")

