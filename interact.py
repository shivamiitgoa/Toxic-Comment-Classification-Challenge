target_name = 'obscene'
target = Y_train[target_name]
BadXTrain = X_train_P[target == 1]
BadVectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (1,1))
BadTFIDF = BadVectorizer.fit_transform(BadXTrain)
BadColumnSum = np.array(BadTFIDF.sum(axis=0).flat)
BadSortedIndices = np.argsort(BadColumnSum)
BadFeatures = BadVectorizer.get_feature_names()
BadCommonWords = [BadFeatures[i] for i in BadSortedIndices[-2000:]]
BadCommonWordsSet2000 = set(BadCommonWords)


GoodXTrain = X_train_P[target == 0]
GoodVectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (1,1))
GoodTFIDF = GoodVectorizer.fit_transform(GoodXTrain)
GoodColumnSum = np.array(GoodTFIDF.sum(axis=0).flat)
GoodSortedIndices = np.argsort(GoodColumnSum)
GoodFeatures = GoodVectorizer.get_feature_names()
GoodCommonWords = [GoodFeatures[i] for i in GoodSortedIndices[-5000:]]
GoodCommonWordsSet5000 = set(GoodCommonWords)


engineered_features = list(BadCommonWordsSet2000 - GoodCommonWordsSet5000)
print("Unique features:")
print(engineered_features)
print("Length of features are %d" % len(engineered_features))

from sklearn.feature_extraction.text import CountVectorizer

globalVectorizer = CountVectorizer(vocabulary=engineered_features)

# globalVectorizer = TfidfVectorizer(vocabulary=engineered_features)

X_train_TF = globalVectorizer.fit_transform(X_train_P)
X_test_TF = globalVectorizer.transform(X_test_P)

model = GradientBoostingClassifier()

utils.create_model_and_evaluate(X_train_TF, Y_train[target_name], X_test_TF, Y_test[target_name], model)

