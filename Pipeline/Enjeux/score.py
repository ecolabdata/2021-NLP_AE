#%%
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def scores(y_pred,y_true,labels):
    reslabel = {}
    conf = multilabel_confusion_matrix(y_true,y_pred)
    for label,mat in zip(labels,conf):
        TN = mat[0,0]
        FN = mat[1,0]
        FP = mat[0,1]
        TP = mat[1,1]
        Acc = (TP+TN)/(TP+TN+FP+FP)
        Pre = TP/(TP+FP)
        Rec = TP/(FN+TP)
        F1 = 2*Pre*Rec/(Pre+Rec)
        if TP==0:
            Pre,Rec,F1 = 0,0,0
        reslabel[label] = [Acc,Pre,Rec,F1]
        print(label, reslabel[label])
    return(reslabel)

def hotgrid_score(labels,results,col = 'hot'):
    names = ['Accuracy','Precision','Recall','F1']
    resarray = []
    for lab in labels:
        resarray.append(results[lab])
    # Setting the labels of x axis.
    # set the xticks as student-names
    # rotate the labels by 90 degree to fit the names
    plt.xticks(ticks=np.arange(len(names)),labels=names,rotation=90)
    # Setting the labels of y axis.
    # set the xticks as subject-names
    plt.yticks(ticks=np.arange(len(labels)),labels=labels)
    # use the imshow function to generate a heatmap
    # cmap parameter gives color to the graph
    # setting the interpolation will lead to different types of graphs
    hm =plt.imshow(resarray, cmap=col,interpolation="nearest")
    
    plt.colorbar()
    plt.show()

def sampledistrib(y,labels = None):
    vals = y.sum(axis=0)
    try:
        vals = vals.tolist()[0]
    except:
        pass
    if labels:
        plt.bar(x = [k for k in range(len(vals))],height = vals,tick_label= labels,rotation = 90)
        plt.show()
    else:
        print('Provide labels')


def hotgrid_corr(labels,y):
    names = labels
    resarray = np.corrcoef(y)
    # Setting the labels of x axis.
    # set the xticks as student-names
    # rotate the labels by 90 degree to fit the names
    plt.xticks(ticks=np.arange(len(names)),labels=names,rotation=90)
    # Setting the labels of y axis.
    # set the xticks as subject-names
    plt.yticks(ticks=np.arange(len(labels)),labels=labels)
    # use the imshow function to generate a heatmap
    # cmap parameter gives color to the graph
    # setting the interpolation will lead to different types of graphs
    hm =plt.imshow(resarray, cmap='hot',interpolation="nearest")
    plt.colorbar()
    plt.show()
    

# %%
