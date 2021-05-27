
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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
        reslabel[label] = [Acc,Pre,Rec,F1]
    return(reslabel)

def hotgrid(labels,results):
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
    hm =plt.imshow(resarray, cmap='hot',interpolation="nearest")
    
    plt.colorbar()
