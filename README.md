# SFA4DL-Embedding

This is the repository for the paper: "Making Time Series Embeddings More Interpretable in Deep Learning: Extracting Higher-Level Features via Symbolic Approximation Representations"

It provides examples on how SFA and SAX can be used as DL embedding for a Transformer model on time series data.

### Data

All univariate UCR & UEA datasets are supported.

### Dependencies installation guide

A list of all needed dependencies (other versions can work but are not guaranteed to do so): <br>

Requirements for the installation guide (other version might or might not work as well):

- python==3.9.16
- tensorflow-gpu==2.4.1
- tslearn==0.5.2
- pyts==0.12.0
- seml==0.3.7


To use the jupyter notebook: <br>
pip install --upgrade ipykernel jupyter notebook pyzmq


### How to run

We have two options to run the experiment. Either just test out single configurations with an anaconda notebook or test out multiple parameter combinations over SEML experiments.

#### Just for testing

1. Go to SFATester.ipynb
2. Change parameters
3. Have fun

#### Multiple experiment settings with SEML

1. Set up seml with seml configure <font size="6">(yes you need a mongoDB server for this and yes the results will be saved a in separate file, however seml does a really well job in managing the parameter combinations in combination with slurm) </font>
2. Configure the yaml file you want to run. Probably you only need to change the number of maximal parallel experiments ('experiments_per_job' and 'max_simultaneous_jobs') and the memory and cpu use ('mem' and 'cpus-per-task').
3. Add and start the seml experiment. For example like this:
	1. seml sfaModel add sfaModel.yaml
	2. seml sfaModel start
4. Check with "seml sfaModel status" till all your experiments are finished 
5. Please find the results in the results folder. It includes a dict which can be further processed with the code in sfaresultprocessing.py

## Cite and publications

This code represents the used model for the following publication:<br>
"Making Time Series Embeddings More Interpretable in Deep Learning: Extracting Higher-Level Features via Symbolic Approximation Representations" (TODO Link)


If you use, build upon this work or if it helped in any other way, please cite the linked publication.
