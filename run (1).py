import pickle
import numpy as np

from blend import *
from helpers import load_data
from utils import get_indices
import nnmf.split_data

def mf_pred(user_features, item_features, input_path,
                          verbose=False):
    nnz_train = np.array(pd.read_csv(input_path, sep='\t', header=None,))
    mf = []
    for i, k in enumerate(nnz_train):
        item = int(k[0])-1
        user = int(k[1])-1
        mf.append( user_features[:, user].T.dot(item_features[:, item]))
    return mf


# load training set
print('Loading training set')
train = load_data('../data/data_train.csv')
nnz_train = get_indices(train)


"""
After some tests, the best score is achieved by only blending the als predictions (that means dilating them).
Another good performance was achieved by blending ALS, user-based  and item based collaborative filtering. Details about 
the tests can be found in the notebook 'src/notebook/blending.ipynb'.
"""

# Train the model (actually if the files are present, nothing is done.
print('Training the model')
_, user_features, item_features = matrix_factorization_als(ratings, None, verbose=True, stop_criterion=0.00001,
                                                           lambda_user=0.014, lambda_item=0.575,
                                                           num_features=20)
write_als_prediction(user_features, item_features)


als_predictions = nnmf.split_data.load_data2('./data/predictions/als_prediction.csv')

# Train the blending with Ridge Regression
# print('Training the blending with ridge regression')
alpha = 0.1
re_als = als_pred(user_features, item_features, input_path='../data/mov_kaggle.all' )
label = pd.read_csv('./data/mov_kaggle.all', sep='\t', header=None).iloc[:,2]
w, rm= blend_train(als = np.array(re_als).reshape((len(re_als),1)), alpha=0.1, label=label)



# Apply weight vector to final predictions (and not to training ratings anymore)
print('Blending the predictions')
m_sub, _ = build_matrix(nnz_sub, methods, None, None, None, None, None, als_predictions, None, None, predict=True)
y_predict_sub = m_sub.dot(w)

for i in range(len(y_predict_sub)):
    y_predict_sub[i] = min(5, max(1, y_predict_sub[i]))

# Write prediction to file
path = '../data/predictions/blended_prediction.csv'
print("Writing prediction to file : '{}'".format(path))
with open(path, 'w') as output:
    output.write('Id,Prediction\n')
    for i, (row, col) in enumerate(nnz_sub):
        output.write('r{}_c{},{}\n'.format(row + 1, col + 1, y_predict_sub[i]))
