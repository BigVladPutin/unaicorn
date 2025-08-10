def prediction_model(pclass,sex,age,sibsp,parch,fare,embarked,title):
    import pickle
    import os
    x = [[pclass,sex,age,sibsp,parch,fare,embarked,title]]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'titanic_model.sav')
    randomforest = pickle.load(open(model_path, 'rb'))
    #randomforest = pickle.load(open('titanic_model.sav', 'rb'))
    predictions = randomforest.predict(x)
    if predictions == 0:
        predictions = 'Not survived'
    elif predictions == 1:
        predictions = 'Survived'
    return predictions
