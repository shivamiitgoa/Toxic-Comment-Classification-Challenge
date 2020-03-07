import os
import random 
import sys 
import pickle 
import json 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone

import utils 

random.seed(0)
np.random.seed(0)
sys.setrecursionlimit(10000)

# ------------------------------------- Loading dataset ---------------------------------------------------------

train_set = pd.read_csv("dataset/train.csv")
test_set = pd.read_csv("dataset/test.csv")
sample_submission = pd.read_csv("dataset/sample_submission.csv")

target_column_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

X = train_set['comment_text'] 
Y = train_set[target_column_names] 

#------------------------------------ Splitting train-test set -------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42) 

#------------------------------------ Converting comments into vectors ----------------------------------------
# Convert all the letters to lowercase 
# Remove all the punctuations 
# Stemmed all the words 
# Converted each document into a vector 

X_train_P_Pickle = 'X_train_P.pickle'
X_test_P_Pickle = 'X_test_P.pickle' 
if os.path.isfile(X_train_P_Pickle) and os.path.isfile(X_test_P_Pickle):
    X_train_P = pickle.load(open(X_train_P_Pickle, 'rb'))
    X_test_P = pickle.load(open(X_test_P_Pickle, 'rb'))
else: 
    utils.log("Preprocessing...")
    X_train_P = X_train.map(utils.commentPreprocessor)
    X_test_P = X_test.map(utils.commentPreprocessor)
    pickle.dump(X_train_P, open(X_train_P_Pickle, 'wb'))
    pickle.dump(X_test_P, open(X_test_P_Pickle, 'wb'))

vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (1,1))
X_train_TFIDF_pickle = "X_train_TFIDF.pickle"
X_test_TFIDF_pickle = "X_test_TFIDF.pickle"

if os.path.isfile(X_train_TFIDF_pickle) and os.path.isfile(X_test_TFIDF_pickle):
    X_train_TFIDF = pickle.load(open(X_train_TFIDF_pickle, 'rb'))
    X_test_TFIDF = pickle.load(open(X_test_TFIDF_pickle, 'rb'))
else:
    utils.log("Converting to vectors...")
    X_train_TFIDF = vectorizer.fit_transform(X_train_P)
    X_test_TFIDF = vectorizer.transform(X_test_P)

    # # Reduce dimension using SVD
    # utils.log("Reducing dimension...")
    # from sklearn.decomposition import TruncatedSVD
    # svd = TruncatedSVD(n_components=1000, random_state=0)
    # X_train_TFIDF = svd.fit_transform(X_train_TFIDF)
    # X_test_TFIDF = svd.transform(X_test_TFIDF)
    # utils.log("Dimension reduced...")

    pickle.dump(X_train_TFIDF, open(X_train_TFIDF_pickle, 'wb'))
    pickle.dump(X_test_TFIDF, open(X_test_TFIDF_pickle, 'wb'))


debug = True
global_results = {}


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

models = {
    # "GradientBoostingClassifier()" : GradientBoostingClassifier(),
    # "DecisionTreeClassifier(random_state=0)": DecisionTreeClassifier(random_state=0),
    # "KNeighborsClassifier(3)": KNeighborsClassifier(3),
    # "SVC(kernel=\"linear\", C=0.025, random_state=0)": SVC(kernel="linear", C=0.025, random_state=0),
    # "SVC(gamma=2, C=1, random_state=0)": SVC(gamma=2, C=1, random_state=0),
    # "GaussianProcessClassifier(1.0 * RBF(1.0), random_state=0)": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=0),
    # "RandomForestClassifier(random_state=0)": RandomForestClassifier(random_state=0),
    # "MLPClassifier(alpha=1, max_iter=1000)": MLPClassifier(alpha=1, max_iter=1000),
    # "AdaBoostClassifier(random_state=0)": AdaBoostClassifier(random_state=0),
    # "GaussianNB()": GaussianNB(),
    # "QuadraticDiscriminantAnalysis()": QuadraticDiscriminantAnalysis(),
    # "ExtraTreesClassifier(random_state=0)": ExtraTreesClassifier(random_state=0),
    "LogisticRegression(random_state=0, solver='lbfgs')": LogisticRegression(random_state=0, solver='lbfgs')
}


for model_name in models:
    if debug: utils.log("------------------ Model : %s -----------------------" % model_name)
    try:
        global_results[model_name] = {}
        scores = []
        for target in target_column_names:
            model = clone(models[model_name])
            target_result = utils.create_model_and_evaluate(X_train_TFIDF, Y_train[target], X_test_TFIDF, Y_test[target], model)
            global_results[model_name][target] = target_result
            scores.append(target_result['AUC'])
        global_results[model_name]['Mean AUC'] = np.mean(scores)
        if debug: print(json.dumps(global_results[model_name], indent=2))
    except Exception as e:
        print(e)


json.dump(global_results, open('log.json', 'w'), indent=4)

