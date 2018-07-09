import os
import pandas as pd 
import numpy as np
from collections import Counter
import plot
import model

def load_data(filename):
    filename = os.path.join(model.BASE_PATH,filename)
    data = pd.read_csv(filename)
    return data

def detect_outliers(df,n,features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

def analysis_feature(train_data,dataset):
    #relation
    # plot.plot_heatmap(train_data[["Survived","SibSp","Parch","Age","Fare"]],'Relation.png')
    #SibSp
    # plot.plot_factor(x='SibSp',y='Survived',data=train_data,savename='SibSp.png')
    #Parch
    # plot.plot_factor(x='Parch',y='Survived',data=train_data,savename='Parch.png')
    #Age
    # plot.plot_faceGrid(train_data,'Age.png',)
    # plot.plot_age_distribution(train_data,'Age_2.png')
    #Fare
    print(dataset["Fare"].isnull().sum())
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    raw_input()
    # plot.plot_dict(dataset,color='m',savename='Fare.png')
    dataset['Fare'] = dataset['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
    # plot.plot_dict(dataset,color='b',savename='Fare_2.png')
    #Sex
    # plot.plot_bar(x='Sex',y='Survived',data=train_data,savename='Sex.png')
    #Pclass
    # plot.plot_factor(x='Pclass',y='Survived',data=train_data,savename='Pclass.png')
    #Embarked
    print(dataset["Embarked"].isnull().sum())
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    raw_input()
    # plot.plot_factor(x='Embarked',y='Survived',data=train_data,savename='Embarked.png')
    # plot.plot_factor(x='Embarked',y='Survived',data=train_data,savename='Embarked_2.png',y_lable='Count',kind='count')
    return dataset

def fill_age_values(train_data,dataset):
    # plot.plot_box(y='Age',x='Sex',data=dataset,savename='Age_Sex.png')
    # plot.plot_box(y='Age',x='Sex',hue='Pclass',data=dataset,savename='Age_Pclass.png')
    # plot.plot_box(y='Age',x='Parch',data=dataset,savename='Age_Parch.png')
    # plot.plot_box(y='Age',x='SibSp',data=dataset,savename='Age_SibSp.png')
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
    # plot.plot_heatmap(data=dataset[["Age","Sex","SibSp","Parch","Pclass"]],savename='Age_Relation.png',camp='BrBG')
    
    index_NaN_age = list(dataset['Age'][dataset['Age'].isnull()].index)
    for i in index_NaN_age :
        age_med = dataset['Age'].median()
        age_pred = dataset['Age'][((dataset['SibSp'] == dataset.iloc[i]['SibSp']) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred) :
            dataset['Age'].iloc[i] = age_pred
        else :
            dataset['Age'].iloc[i] = age_med
    # plot.plot_box(x='Survived',y='Age',data=train_data,savename='Age_Survival_1.png')
    # plot.plot_box(x='Survived',y='Age',data=train_data,kind='violin',savename='Age_Survival_2.png')
    return dataset

def build_title(dataset):
    print(dataset['Name'].head())
    raw_input()
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Title"] = pd.Series(dataset_title)
    # print(dataset["Title"].head())
    # plot.plot_count(x='Title',data=dataset,savename='Title.png')
    dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    dataset["Title"] = dataset["Title"].astype(int)
    # plot.plot_count(x='Title',data=dataset,xticklabels=["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"],savename='Title_2.png')
    # plot.plot_factor(x='Title',y='Survived',data=dataset,savename='Title_3.png')
    dataset.drop(labels = ["Name"], axis = 1, inplace = True)
    return dataset

def build_family_size(dataset):
    dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
    # plot.plot_factor(x='Fsize',y='Survived',data=dataset,savename='Fsize.png')

    dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

    # plot.plot_factor(x='Single',y='Survived',data=dataset,savename='Fsize_2.png')
    # plot.plot_factor(x='SmallF',y='Survived',data=dataset,savename='Fsize_3.png')
    # plot.plot_factor(x='MedF',y='Survived',data=dataset,savename='Fsize_4.png')
    # plot.plot_factor(x='LargeF',y='Survived',data=dataset,savename='Fsize_5.png')   

    dataset = pd.get_dummies(dataset, columns = ["Title"])
    dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
    return dataset

def build_carbin(dataset):
    print(dataset["Cabin"].head())
    raw_input()
    # print(dataset["Cabin"].describe())
    # print(dataset["Cabin"].isnull().sum())
    # print(dataset["Cabin"][dataset["Cabin"].notnull()].head())
    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
    # plot.plot_count(x='Carbin',data=dataset,order=['A','B','C','D','E','F','G','T','X'])
    # plot.plot_factor(x='Carbin',y='Survived',data=dataset,order=['A','B','C','D','E','F','G','T','X'],savename='Carbin_2.png')
    dataset = pd.get_dummies(dataset, columns=['Cabin'],prefix='Cabin')
    return dataset

def build_ticket(dataset):
    print(dataset["Ticket"].head())
    raw_input()
    Ticket = []
    for i in list(dataset.Ticket):
        if not i.isdigit() :
            Ticket.append(i.replace('.','').replace('/','').strip().split(' ')[0])
        else:
            Ticket.append("X")
            
    dataset['Ticket'] = Ticket
    # print(dataset["Ticket"].head())
    dataset = pd.get_dummies(dataset,columns=['Ticket'],prefix='T')
    return dataset

def rebuild_feature(dataset):
    dataset = build_title(dataset)
    dataset = build_family_size(dataset)
    dataset = build_carbin(dataset)
    dataset = build_ticket(dataset)
    # Create categorical values for Pclass
    dataset["Pclass"] = dataset["Pclass"].astype("category")
    dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
    dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
    return dataset

if __name__ == '__main__':
    train_data = load_data('data/train.csv')
    test_data = load_data('data/test.csv')
    IDtest = test_data['PassengerId']
    print(train_data.sample(5))
    raw_input()

    #remove outliers
    outliers_to_drop = detect_outliers(train_data,2,["Age","SibSp","Parch","Fare"])
    print(train_data.loc[outliers_to_drop])
    raw_input()
    train_data = train_data.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
    train_len = len(train_data)

    dataset = pd.concat(objs=[train_data,test_data],axis=0).reset_index(drop=True)
    dataset = dataset.fillna(np.nan)
    print(dataset.isnull().sum())
    raw_input()

    dataset = analysis_feature(train_data,dataset)
    dataset = fill_age_values(train_data,dataset)
    # Feature
    dataset = rebuild_feature(dataset)

    train = dataset[:train_len]
    test = dataset[train_len:]
    test.drop(labels=['Survived'],axis = 1,inplace=True)

    train['Survived'] = train['Survived'].astype(int)
    y_train = train['Survived']
    X_train = train.drop(labels=['Survived'],axis=1)

    # cv_res,cv_std = model.cross_val(X=X_train,y=y_train)
    # plot.plot_model_accuracy(data=cv_res,std=cv_std,savename='model_accuracy.png')

    # model.train_model(X_train,y_train)
    # model.eval_features(X_train,y_train)
    model.train(X_train,y_train)
    model.predict(test,IDtest)