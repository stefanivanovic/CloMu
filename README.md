# CloMu
Modeling Cancer Evolution With Reinforcement Learning

Dependencies:
pytorch, numpy. 
If one wants to using the plotting functions and some evaluation functions, then also matplotlib and scipy. 

Before running the code, make sure to unzip any required zipped folders. 



### Running on the default breast cancer and leukemia data sets:

To train a model on the breast cancer data set or leukemia data set run
"python cancer.py real breast train"
or 
"python cancer.py real leukemia train". 


To plot the results of these models run
"python cancer.py real breast plot".
or
"python cancer.py real leukemia plot"

To save the predicted trees for these models run
"python cancer.py real breast predict"
or
"python cancer.py real leukemia predict".


To determine the proportion based fitness of different mutations in the leukemia data set run 
"python cancer.py real proportion".




### RECAP Simulated Data:


To train a model on the recap data sets with 5 mutations or 7 mutations run
"python cancer.py recap m5 train"
or
"python cancer.py recap m7 train".

To evaluate a model on the recap data sets run
"python cancer.py recap m5 evaluate"
or 
"python cancer.py recap m7 evaluate".


To plot the results of evaluating the model run
"python cancer.py recap m5 plot"
or 
"python cancer.py recap m7 plot".


### Additional Simulated Expirements:


To train a model on the simulated data set of binary causal connections run
"python3 cancer.py test causal train".

To print the results of evaluating this model run
"python3 cancer.py test causal print".


To train a model on the simulated data set of evolutionary pathways run
"python3 cancer.py test pathway train".

To save the predicted pathways of the model run
"python3 cancer.py test pathway evaluate".

To print the resulting accuracy of the model run
"python3 cancer.py test pathway print".

### Using the code on new data sets:

First, save your data set a a list of lists of trees, with one list of tree per patient. 
Specifically, save this in a .npy file which can done with "data = np.array(data, dtype=object)" and "np.save('./dataNew/customData/' + dataName + '.npy', data)". In that line, dataName is the name of your data set and "./dataNew/customData/" is the location in which it should be placed. 

To train a model on the new data set run
"python cancer.py custom dataName train maxM trainPercent"
where dataName is the data set name, maxM is the maximum number of mutations per patient you want to allow and trainPercent is the proportion of the data that should be used for training instead of testing, and can be set to 1.0 if one doesn't want a seperate test set. 

To plot the results of the model run 
"python cancer.py custom dataName plot".

To save the predictions of the model run
"python cancer.py custom dataName predict".

The predicts will be saved as a list of predictions for each patient. For each patient, the data is a list containing the patient number, the array of probabilities for each tree, and the list of possible trees corresponding to those probabilities. This file will be ./dataNew/predictedTrees_dataName.npz, where dataName is the name of the data set. 






