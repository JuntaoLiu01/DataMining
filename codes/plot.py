import matplotlib.pyplot as plt
import seaborn as sns 

def plot_heatmap(data,savename=None,camp='coolwarm'):
    sns.heatmap(data.corr(),annot=True,cmap=camp)
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

def plot_factor(x,y,data,savename=None,order=None,y_lable='Survival Probability',kind='bar',size=6,palette='muted'):
    if order is not None:
        g = sns.factorplot(x=x,y=y,data=data,order=order,kind=kind,size=6,palette=palette)
    else:
        g = sns.factorplot(x=x,y=y,data=data,kind=kind,size=6,palette=palette)

    g.despine(left=True)
    g = g.set_ylabels(y_lable)
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

def plot_faceGrid(data,savename=None,col='Survived',feat='Age'):
    g = sns.FacetGrid(data,col=col)
    g = g.map(sns.distplot,feat)
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

def plot_age_distribution(data,savename=None):
    g = sns.kdeplot(data['Age'][(data['Survived'] == 0) & (data['Age'].notnull())], color='Red', shade = True)
    g = sns.kdeplot(data['Age'][(data['Survived'] == 1) & (data['Age'].notnull())], ax =g, color='Blue', shade= True)
    g.set_xlabel('Age')
    g.set_ylabel('Frequency')
    g = g.legend(['Not Survived','Survived'])
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

def plot_dict(data,color,savename=None,label='Skewness:%.2f'):
    g = sns.distplot(data['Fare'], color=color,label=label%(data['Fare'].skew()))
    g = g.legend(loc="best")
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

def plot_bar(x,y,data,savename=None,y_lable='Survival Probability'):
    g = sns.barplot(x=x,y=y,data=data)
    g = g.set_ylabel(y_lable)
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

def plot_box(x,y,data,savename=None,hue=None,kind='box'):
    if hue is not None:
        g = sns.factorplot(y=y,x=x,data=data,hue=hue,kind=kind)
    else:
        g = sns.factorplot(y=y,x=x,data=data,kind=kind)
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

def plot_count(x,data,xticklabels=None,order=None,savename=None):
    if order is not None:
        g = sns.countplot(x=x,data=data,order=order)
    else:
        g = sns.countplot(x=x,data=data)

    if xticklabels is not None:
        g = plt.setp(g.get_xticklabels(),rotation=45)
    else:
        g.set_xticklabels(xticklabels)

    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

def plot_model_accuracy(data,std,savename=None):
    g = sns.barplot("CrossValMeans","Algorithm",data=data, palette="Set3",orient = "h",**{'xerr':std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight')
    else:
        plt.show()
    plt.close()