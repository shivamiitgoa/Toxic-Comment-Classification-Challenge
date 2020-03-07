from datetime import datetime
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


stemmer = PorterStemmer()


def commentPreprocessor(comment):
    '''
    Converts the string given into lowercase and removes all characters which are not alphabets.
    '''
    comment_lower = comment.lower()
    comment_alphabets = []
    for c in comment_lower:
        if c.isalpha():
            comment_alphabets.append(c)
        else:
            comment_alphabets.append(' ')
    comment_alphabets = "".join(comment_alphabets)
    comment_words = comment_alphabets.split()
    stemmed_comment_words = [stemmer.stem(x) for x in comment_words if len(x) >= 3]
    return " ".join(stemmed_comment_words)


def log(message):
    '''
    Prints the given message along with the current time
    '''
    now = datetime.now()
    time_string = now.strftime("%H:%M:%S:%f")
    print("%s: %s" % (time_string, message))


def create_model_and_evaluate(XTrainArg, Y_train, XTestArg, Y_test, model, debug=False):
    '''
    Trains the given model, tests it and return its score.
    '''
    if debug: log("Training model...")
    lr_model = model.fit(XTrainArg, Y_train)
    if debug: log("Making predictions...")
    predictions = lr_model.predict(XTestArg)
    prediction_prob = lr_model.predict_proba (XTestArg)
    score = roc_auc_score(Y_test, prediction_prob[:,1])
    _confusion_matrix = confusion_matrix(Y_test, predictions)
    TN, FP = _confusion_matrix[0]
    FN, TP = _confusion_matrix[1]
    TN, FP, FN, TP = int(TN), int(FP), int(FN), int(TP)
    Recall = TP / (TP + FN)
    results = {
        "TN": TN,
        "FP": FP, 
        "FN": FN,
        "TP": TP,
        "Recall": Recall,
        "AUC": score
    }
    return results

