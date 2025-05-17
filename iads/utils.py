# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2025

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 

def genere_dataset_uniform(d, nc, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        d: nombre de dimensions de la description
        nc: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data = np.random.uniform(binf,bsup,(2*nc,d))
    etiquettes = np.array([-1 for i in range(0,nc)] + [+1 for i in range(0,nc)])
    
    return (data,etiquettes)

#######################################################################################

def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nc):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    x = np.random.multivariate_normal(negative_center,negative_sigma,nc)
    y = np.random.multivariate_normal(positive_center,positive_sigma,nc)
    label = np.array([-1 for i in range(0,nc)] + [+1 for i in range(0,nc)])
    return (np.concatenate((x,y), axis=0),label)

#######################################################################################

def plot2DSet(desc,labels,nom_dataset= "Dataset", avec_grid=False):    
    """ ndarray * ndarray * str * bool-> affichage
        nom_dataset (str): nom du dataset pour la légende
        avec_grid (bool) : True si on veut afficher la grille
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    data_negatifs = desc[labels == -1]
    data_positifs = desc[labels == +1]
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="red", label='classe -1') # 'o' rouge pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="blue", label='classe +1') # 'x' bleu pour la classe +1
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    if avec_grid:
        plt.grid()
    plt.show()

#######################################################################################

def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

#######################################################################################

def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    # Centres des 4 nuages de points (coins du plan)
    centers = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])  # Coins haut gauche, haut droit, bas gauche, bas droit
    
    # Initialisation des tableaux pour les points (X) et labels (y)
    X = np.zeros((4 * n, 2))
    y = np.zeros(4 * n)
    
    # Attribution des classes :
    # Classe 1 pour les coins haut gauche (0,1) et bas droit (1,0)
    # Classe -1 pour les coins haut droit (1,1) et bas gauche (0,0)
    class_assignments = [1, -1, -1, 1]
    
    # Génération des points pour chaque nuage
    for i, center in enumerate(centers):
        # Générer des points autour du centre avec la variance spécifiée
        points = np.random.normal(center, var, (n, 2))
        
        # Insérer les points générés dans X et les labels dans y
        X[i * n:(i + 1) * n] = points
        y[i * n:(i + 1) * n] = class_assignments[i]
    
    return X, y

