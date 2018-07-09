from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.externals import joblib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

kfold = StratifiedKFold(n_splits=10)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_PATH,'models')

def init_model():
    random_state = 2
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(LinearDiscriminantAnalysis())
    return classifiers

def cross_val(X,y):
    classifiers = init_model()
    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier,X=X,y=y,scoring='accuracy',cv=kfold,n_jobs=1))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({'CrossValMeans':cv_means,'CrossValerrors': cv_std,'Algorithm':['SVC','DecisionTree','AdaBoost',
    'RandomForest','ExtraTrees','GradientBoosting','MultipleLayerPerceptron','KNeighboors','LogisticRegression','LinearDiscriminantAnalysis']})
    return cv_res,cv_std
    
def adaBoost(X,y):
    DTC = DecisionTreeClassifier()
    adaDTC = AdaBoostClassifier(DTC, random_state=7)
    ada_param_grid = {'base_estimator__criterion' : ['gini', 'entropy'],
                'base_estimator__splitter' :   ['best', 'random'],
                'algorithm' : ['SAMME','SAMME.R'],
                'n_estimators' :[1,2],
                'learning_rate':  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
    gsadaDTC = GridSearchCV(adaDTC,param_grid=ada_param_grid,cv=kfold,scoring='accuracy',n_jobs=2,verbose=1)
    gsadaDTC.fit(X,y)
    ada_best = gsadaDTC.best_estimator_
    print(gsadaDTC.best_score_)
    modelname = os.path.join(MODEL_DIR,'Ada.m')
    joblib.dump(ada_best,modelname)
    return ada_best

def extraTrees(X,y):
    ExtC = ExtraTreesClassifier()
    ex_param_grid = {'max_depth': [None],
                'max_features': [1, 3, 10],
                'min_samples_split': [2, 3, 10],
                'min_samples_leaf': [1, 3, 10],
                'bootstrap': [False],
                'n_estimators' :[100,300],
                'criterion': ['gini']}
    gsExtC = GridSearchCV(ExtC,param_grid=ex_param_grid,cv=kfold,scoring='accuracy',n_jobs=2,verbose=1)
    gsExtC.fit(X,y)
    ExtC_best = gsExtC.best_estimator_
    print(gsExtC.best_score_)
    modelname = os.path.join(MODEL_DIR,'Ext.m')
    joblib.dump(ExtC_best,modelname)
    return ExtC_best

def randomForest(X,y):
    RFC = RandomForestClassifier()
    rf_param_grid = {'max_depth': [None],
                'max_features': [1, 3, 10],
                'min_samples_split': [2, 3, 10],
                'min_samples_leaf': [1, 3, 10],
                'bootstrap': [False],
                'n_estimators' :[100,300],
                'criterion': ['gini']}
    gsRFC = GridSearchCV(RFC,param_grid =rf_param_grid,cv=kfold,scoring='accuracy',n_jobs=2,verbose = 1)
    gsRFC.fit(X,y)
    RFC_best = gsRFC.best_estimator_
    print(gsRFC.best_score_)
    modelname = os.path.join(MODEL_DIR,'RFC.m')
    joblib.dump(RFC_best,modelname)
    return RFC_best

def gradientBoosting(X,y):
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss' : ['deviance'],
                'n_estimators' : [100,200,300],
                'learning_rate': [0.1, 0.05, 0.01],
                'max_depth': [4, 8],
                'min_samples_leaf': [100,150],
                'max_features': [0.3, 0.1] 
                }
    gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring='accuracy', n_jobs=2, verbose = 1)
    gsGBC.fit(X,y)
    GBC_best = gsGBC.best_estimator_
    print(gsGBC.best_score_)
    modelname = os.path.join(MODEL_DIR,'GBC.m')
    joblib.dump(GBC_best,modelname)
    return GBC_best

def supportVector(X,y):
    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'], 
                    'gamma': [ 0.001, 0.01, 0.1, 1],
                    'C': [1, 10, 50, 100,200,300, 1000]}
    gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring='accuracy',n_jobs=2, verbose = 1)
    gsSVMC.fit(X,y)
    SVMC_best = gsSVMC.best_estimator_
    print(gsSVMC.best_score_)
    modelname = os.path.join(MODEL_DIR,'SVM.m')
    joblib.dump(SVMC_best,modelname)
    return SVMC_best

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=-1,train_sizes=np.linspace(.1, 1.0, 5),savename=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')

    plt.legend(loc='best')
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    # return plt   

def train_model(X,y,plot=True):
    ada_best = adaBoost(X,y)
    ExtC_best = extraTrees(X,y)
    RFC_best = randomForest(X,y)
    GBC_best = gradientBoosting(X,y)
    SVMC_best = supportVector(X,y)
    if plot:
        plot_learning_curve(RFC_best,'RF mearning curves',X,y,cv=kfold,savename='RFC.png')
        plot_learning_curve(ExtC_best,'ExtraTrees learning curves',X,y,cv=kfold,savename='Ext.png')
        plot_learning_curve(SVMC_best,'SVC learning curves',X,y,cv=kfold,savename='SVC.png')
        plot_learning_curve(ada_best,'AdaBoost learning curves',X,y,cv=kfold,savename='Ada.png')
        plot_learning_curve(GBC_best,'GradientBoosting learning curves',X,y,cv=kfold,savename='GBC.png')

def eval_features(X,y):
    # SVMC_best = supportVector(X,y)
    ada_best = adaBoost(X,y)
    ExtC_best = extraTrees(X,y)
    RFC_best = randomForest(X,y)
    GBC_best = gradientBoosting(X,y)
    
    nrows = ncols = 2
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex='all',figsize=(15,15))
    names_classifiers = [('AdaBoosting', ada_best),('ExtraTrees',ExtC_best),('RandomForest',RFC_best),('GradientBoosting',GBC_best)]
    nclassifier = 0
    for row in range(nrows):
        for col in range(ncols):
            name = names_classifiers[nclassifier][0]
            classifier = names_classifiers[nclassifier][1]
            indices = np.argsort(classifier.feature_importances_)[::-1][:40]
            g = sns.barplot(y=X.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
            g.set_xlabel('Relative importance',fontsize=12)
            g.set_ylabel('Features',fontsize=12)
            g.tick_params(labelsize=9)
            g.set_title(name + ' feature importance')
            nclassifier += 1
    plt.savefig('feature_importance.png')

def load_models():
    files = os.listdir(MODEL_DIR)
    models = {}
    for file in files:
        if file.endswith('.m'):
            filename = os.path.join(MODEL_DIR,file)
            models[file[:-2]] = joblib.load(filename)
    return models

def train(X,y):
    models = load_models()
    estimators = []
    for key,value in models.items():
        estimators.append((key,value))
    votingC = VotingClassifier(estimators=estimators, voting='soft',n_jobs=2)
    votingC = votingC.fit(X,y)
    modelname = os.path.join(MODEL_DIR,'model.m')
    joblib.dump(votingC,modelname)
    return votingC

def predict(test,IDtest):
    models = load_models()
    votingC = models['model']
    test_Survived = pd.Series(votingC.predict(test), name='Survived')
    results = pd.concat([IDtest,test_Survived],axis=1)
    results.to_csv('results.csv',index=False)
