# recommender-system
EPFL-ML

### Organization of the folder:
* **source code** :
    * run.py : python script that creates the best prediction achieved.
    * baselines.py : functions for general_mean, user_means and item_means baselines.
    * cross_validation.py : implementation of cross_validation methods in the case of a recommender system.
    * helpers.py, plots.py : provided in the lab10
    * utils.py : useful functions to read files, write predictions to file, initialize matrix factorization methods, 
    and compute prediction error.
    * matrix_factorization.py functions to perform ALS and SGD matrix factorization
    *neural_net.py : implementation of multi-layer-perceptron
    * blending.py : functions to perform the blending of various models
    *NN&SVD.ipynb : Implementation of SVD, SVD++, XgBoost and training curve for MLP
    *MF&KNN&Other : Implementation of KNN-user, KNN-item, RMSE for other models
* **Foleder "data"** : 
    * *data_train.csv* : the traing data downloaded from the crowdAI challenge
    * *sample_submission.csv* : the sample submission file provided by the crowdAI challenge
    * *submit.csv* : the predictions generated by the model as the final result
* **report.pdf** : report of the project


### Libraries Used:
To run the code in this repository, following libraries are required :
* **Surprise** : pip install surprise

* **Keras** : install Tensorflow and then install Keras using pip install keras

* **Scikit Learn** : pip install -U scikit-learn



