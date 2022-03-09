import pickle

try:
    with open("saved-models/SVM_E-I.sav", "rb") as file:
        ei_classifier = pickle.load(file)
    with  open("saved-models/Xgboost_N-S.sav", "rb") as file:
        ns_classifier = pickle.load(file)
    with open("saved-models/SVM_F-T.sav", "rb") as file:
        ft_classifier = pickle.load(file)
    with  open("saved-models/Xgboost_J-P.sav", "rb") as file:
        jp_classifier = pickle.load(file)
except FileNotFoundError:
    print("Model not found!")

try:
    with open("vectorizer/vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    print("Tokenizer not found!")
    
def preprocessing(test, vectorizer, ei_classifier, ns_classifier, ft_classifier, jp_classifier):
    stopword_list = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    test = contractions.fix(test)
    test = test.lower()
    test = re.sub(r'http[s]?://\S+', '', test)
    test = re.sub(r'[^A-Za-z0-9]+', ' ', test)
    test = re.sub(r' +', ' ', test)
    test = " ".join([word for word in test.split() if not len(word) <3])
    test = word_tokenize(test)
    test = [word for word in test if not word in stopword_list]
    test = [lemmatizer.lemmatize(word) for word in test]
    test = " ".join(test)
    test = vectorizer.transform([test])
    
    prediction = ""
    e_or_i = "E" if ei_classifier.predict(test)[0] == 1 else "I"
    n_or_s = "N" if ns_classifier.predict(test)[0] == 1 else "S"
    f_or_t = "F" if ft_classifier.predict(test)[0] == 1 else "T"
    j_or_p = "J" if jp_classifier.predict(test)[0] == 1 else "P"
    prediction = e_or_i + n_or_s + f_or_t + j_or_p
    
    return prediction