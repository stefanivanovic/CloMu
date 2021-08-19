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
"python cancer.py real breast plot"
or
"python cancer.py real leukemia plot"

To save the predicted trees for these models run
"python cancer.py real breast predict"
or
"python cancer.py real leukemia predict"


To determine the proportion based fitness of different mutations in the leukemia data set run 
"python cancer.py real proportion"




### Recap Simulated Data:


To train a model on the recap data sets with 5 mutations or 7 mutations run

python cancer.py recap m5 train

or

python cancer.py recap m7 train


To evaluate a model on the recap data sets run

python cancer.py recap m5 evaluate

or 

python cancer.py recap m7 evaluate


To plot the results of evaluating the model run

python cancer.py recap m5 plot

or 

python cancer.py recap m7 plot



#Additional Simulated Expirements:


To train a model on the simulated data set of binary causal connections run

python3 cancer.py test causal train

To print the results of evaluating this model run

python3 cancer.py test causal print


To train a model on the simulated data set of evolutionary pathways run

python3 cancer.py test pathway train

To save the predicted pathways of the model run

python3 cancer.py test pathway evaluate

To print the resulting accuracy of the model run

python3 cancer.py test pathway print


custom data set:

To train a model on a custom data set run

python cancer.py real breast train

or 

python cancer.py real leukemia train




To plot the results of these models run

python cancer.py real breast plot

or

python cancer.py real leukemia plot




