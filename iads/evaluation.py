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
def crossval(X, Y, n_iterations, iteration):
    start = int(iteration *(X.shape[0]//n_iterations))
    end = int((iteration + 1) *(X.shape[0]//n_iterations))
    Xtest = X[start:end]
    Ytest = Y[start:end]
    Xapp = np.concatenate((X[:start],X[end:]))
    Yapp = np.concatenate((Y[:start],Y[end:]))

    return Xapp, Yapp, Xtest, Ytest

##############################################################################

def crossval_strat(X, Y, n_iterations, iteration):
    #############
    # A COMPLETER
    #############
    classe = np.unique(Y) 
    indices_classe = {c: np.where(Y==c)[0] for c in classe}
    test_indices= []
    train_indices =[]
    for c in classe:
        indices = indices_classe[c]  # Indices de la classe actuelle
        taille_test = len(indices) // n_iterations 
        start =iteration*taille_test
        end=(iteration+1)*taille_test
        test_indices.extend(indices[start:end]) 
        train_indices.extend(np.concatenate((indices[:start], indices[end:]))) 

    Xtest,Ytest=X[test_indices],Y[test_indices]
    Xapp,Yapp=X[train_indices],Y[train_indices]
    return Xapp, Yapp, Xtest, Ytest

##############################################################################

def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    # np.std rend l'écart type
    return (np.mean(L), np.std(L))
    
##############################################################################

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
        print("Itération ",i," : taille base app.= ",Xapp.shape[0]," taille base test= ",Xtest.shape[0]," Taux de bonne classif: ",classifieur.accuracy(Xtest, Ytest))
    print("------ fin affichage validation croisée")
    moy,std = analyse_perfs(perf)
    return (perf,moy,std)
    
##############################################################################

# ------------------------ 

