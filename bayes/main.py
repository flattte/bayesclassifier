try:
    import module as md
    import pathhandle as ph
    import numpy as np
    import pandas as pd
    import logs
    import seaborn as sns
    import matplotlib.pyplot as plt
    from random import seed
    import utils
    from utils import split_data, load_csv
except ImportError:
    print("Failed importing modules")
    exit()

    
def np_implementation_iris(): #LEARNS
    data = utils.load_csv(ph.path("iris.csv").path)
    for column in range(4):
        utils.s_to_float(data, column) 
    utils.s_to_int(data, len(data[0])-1) #correct
    
    #fetch data
    train , test = split_data(data, 0.81)
    X_train, X_test, y_train, y_test = utils.label_feature_split_back(train, test, len(train[0]))
    
    #model fitting and predicting test set
    model = md.NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #print results and save them to logs
    model_log = "NaiveBayesClassifier accuracy: {}%".format(model.accuracy(y_test, y_pred)*100)
    print(model_log)
    logs.write_to_file(model_log, "iris.csv")

    #plot of class vs some features
    colnames = ["LP", "sepal lenght", "sepal width", "petalwidth", "class"]
    df = pd.read_csv("../dataset/iris.csv")
    df.columns = colnames
    sns.lmplot('sepal lenght', 'sepal width', data=df, hue='class',fit_reg=False)
    plt.show()


def np_implementation_wine(): #LEARNS 
    data = load_csv('../dataset/wine.data')
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=float(data[i][j])

    #fetch data
    train , test = split_data(data, 0.77)
    X_train, X_test, y_train,  y_test = utils.label_feature_split_front(train, test, 1)

    #model fitting and predicting test set
    model = md.NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #print results and save them to logs
    model_log =("NaiveBayesClassifier accuracy: {}%".format(model.accuracy(y_test, y_pred)*100))
    print(model_log)
    logs.write_to_file(model_log, "wine.data")

    #1,3
    classify = [[14.06,2.15,2.61,17.6,121,2.6,2.51,.31,1.25,5.05,1.06,3.58,1295],
                [12.96,3.45,2.35,18.5,106,1.39,.7,.4,.94,5.28,.68,1.75,675]]
    pred = model.predict(classify)
    print(pred)

def glass_implementation():   #LEARNS
    data = load_csv("../dataset/glass.csv")
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=float(data[i][j])

    #fetch data
    train, test = split_data(data, 0.80)
    X_train, X_test, y_train, y_test = utils.label_feature_split_back(train, test, len(train[0]))

    #model fitting and predicting test set
    model = md.NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #print results and save them to logs
    model_log = "NaiveBayesClassifier accuracy: {}%".format(model.accuracy(y_test, y_pred)*100)
    print(model_log)
    logs.write_to_file(model_log, "glass.csv")

    #2,1
    classify = [[108,1.52365,15.79,1.83,1.31,70.43,0.31,8.61,1.68,0],
                [194,1.51625,13.36,3.58,1.49,72.72,0.45,8.21,0,0]]
    pred = model.predict(classify)
    print(pred)
    
    #plot of class vs some features
    columns= ['LP','RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','class']
    df = pd.read_csv("../dataset/glass.csv")
    df.columns = columns
    sns.lmplot('K', 'Al', data=df, hue='class',fit_reg=False)
    plt.show()


#job stats dataset
def job_implementation(): #NEEDS WORK
    path = ph.path("../dataset/jobdescription.csv")
    df = pd.read_csv(path.path, header=None, sep = ',\s', engine='python')
    print(df.shape)
    #data structure
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    df.columns = col_names
    #print(df.head())
    #print(df.info())
    df['workclass'].replace('?', np.NaN, inplace=True)
    df['occupation'].replace('?', np.NaN, inplace=True)
    df['native_country'].replace('?', np.NaN, inplace=True)
    #todo

if __name__ == '__main__':
    #np_implementation_wine()
    #np_implementation_iris()
    #glass_implementation()
    pass
