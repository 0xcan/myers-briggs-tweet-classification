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
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    print("Tokenizer not found!")
    