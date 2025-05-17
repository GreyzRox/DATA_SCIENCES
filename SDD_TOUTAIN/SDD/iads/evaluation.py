# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 

# crossval:
def crossval(X, Y, n_iterations, iteration):
    Xtest= X[iteration*len(X)//n_iterations:(iteration+1)*len(X)//n_iterations]
    Ytest = Y[iteration*len(Y)//n_iterations:(iteration+1)*len(Y)//n_iterations]
    Xapp = np.concatenate((X[:iteration*len(X)//n_iterations],X[(iteration+1)*len(X)//n_iterations:]))
    Yapp = np.concatenate((Y[:iteration*len(Y)//n_iterations],Y[(iteration+1)*len(Y)//n_iterations:]))
    return Xapp, Yapp, Xtest, Ytest
   
# crossval_strat:
def crossval_strat(X, Y, n_iterations, iteration):
    Xapp = np.empty((0,) + X.shape[1:], dtype=X.dtype)
    Xtest = np.empty((0,) + X.shape[1:], dtype=X.dtype)
    Yapp = np.empty(0, dtype=Y.dtype)
    Ytest = np.empty(0, dtype=Y.dtype)
    
    for classe in np.unique(Y):
        indices_classe = np.where(Y==classe)[0]
        n_classe = len(indices_classe)
        
        taille_test = int(np.ceil(n_classe/n_iterations))

        debut_test = iteration * taille_test
        fin_test = min((iteration+1)*taille_test,n_classe)

        indices_test = indices_classe[debut_test:fin_test]
        indices_train = np.concatenate([indices_classe[:debut_test],indices_classe[fin_test:]])

        Xapp = np.concatenate((Xapp, X[indices_train]))
        Xtest = np.concatenate((Xtest, X[indices_test]))
        Yapp = np.concatenate((Yapp, Y[indices_train]))
        Ytest = np.concatenate((Ytest, Y[indices_test]))
    
    return Xapp, Yapp, Xtest, Ytest
  
# analyse_perfs:
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return (np.mean(L),np.std(L)) 

# validation_croisee:
def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    perf = []
    print("------ affichage validation croisée")
    for i in range(nb_iter):
        classifieur = copy.deepcopy(C)
        Xapp,Yapp,Xtest,Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
        classifieur.train(Xapp, Yapp)
        perf.append(classifieur.accuracy(Xtest, Ytest))
        print("Itération ",i,": taille base app.= ",Xapp.shape[0]," taille base test= ",Xtest.shape[0]," Taux de bonne classif: ",classifieur.accuracy(Xtest, Ytest))
    print("------ fin affichage validation croisée")
    moy,std = analyse_perfs(perf)
    return (perf,moy,std)

# validation_croisee sans print:
def validation_croisee_sans_print(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    perf = []
    for i in range(nb_iter):
        classifieur = copy.deepcopy(C)
        Xapp,Yapp,Xtest,Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
        classifieur.train(Xapp, Yapp)
        perf.append(classifieur.accuracy(Xtest, Ytest))
    moy,std = analyse_perfs(perf)
    return (perf,moy,std)
# ------------------------ 
