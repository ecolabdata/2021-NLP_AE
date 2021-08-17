#%%
import torch
import numpy as np
from Pipeline.Enjeux.multilabel_balancing import make_multilabel_classification

#%%

def optimize_weights(model,loss, epoch = 10):
    import torch
    import numpy as np
    from Pipeline.Enjeux.boosting import F1_loss    
    from torch.utils.data import DataLoader,RandomSampler
    labels = torch.tensor(model.y_sub.to_numpy())
    n = 10
    #x0 = np.random.rand(n)
    #x0 = x0/x0.sum()
    x0 = np.array([1/n]*n)
    x0 = [torch.tensor(w,requires_grad=True) for w in x0]
    sortie = torch.tensor(model.predict(np.matrix(model.X_sub),
                    weights= [w.detach().numpy() for w in x0]).astype(float))
    l = [loss(sortie,labels)]
    optimizer = torch.optim.Adam(x0,lr=0.0005,amsgrad=True)
    losses = [l[0]]
    for e in range(epoch):
        optimizer.zero_grad()
        sortie = torch.tensor(model.predict(np.matrix(model.X_sub),
                    weights= [w.detach().numpy() for w in x0]).astype(float))
        l = loss(sortie,labels)
        l.backward()
        grad = l.grad
        losses.append(t,l.data[0])
        l.requires_grad = True
        xg = []
        for w in x0:
            w.backward()
            xg.append(w.grad)
        #for k in range(len(x0)):
           # x0[k] += xg[k]*
        optimizer.step()

    print(losses)
    return(optimizer,losses)

#%%