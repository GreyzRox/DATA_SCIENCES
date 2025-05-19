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
from iads import Classifiers as classif
import string

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

#######################################################################################

def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = classif.entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = classif.entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)

#######################################################################################

def partitionne(m_desc,m_class,n,s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    return ((m_desc[m_desc[:,n]<=s], m_class[m_desc[:,n]<=s]), \
            (m_desc[m_desc[:,n]>s], m_class[m_desc[:,n]>s]))

#######################################################################################

def nettoyage(chaine):

    res = chaine.lower()
    for car in string.punctuation:
        if(car != "'" and car in chaine):
            res=res.replace(car,' ')
    return res

#######################################################################################

def text2vect(chaine,mots_inutiles):
    chaine_nettoye = nettoyage(chaine)
    res = chaine_nettoye.split()
    return [mot for mot in res if mot not in mots_inutiles]

#######################################################################################

def df2array(df,index_mots):
    mot_to_index = {mot: i for i, mot in enumerate(index_mots)}
    res = []

    for mots in df["les_mots"]:
        v = np.zeros(len(index_mots), dtype = int)
        for mot in mots:
            if mot in mot_to_index:
                v[mot_to_index[mot]] = 1

        res.append(v)

    return np.array(res)

#######################################################################################

def df2array_comptage(df,index_mots):
    mot_to_index = {mot: i for i, mot in enumerate(index_mots)}
    res = []

    for mots in df["les_mots"]:
        v = np.zeros(len(index_mots), dtype = int)
        for mot in mots:
            if mot in mot_to_index:
                v[mot_to_index[mot]] += 1

        res.append(v)

    return np.array(res)

#######################################################################################

def df2array_freq(df,index_mots):
    mot_to_index = {mot: i for i, mot in enumerate(index_mots)}
    res = []

    for mots in df["les_mots"]:
        v = np.zeros(len(index_mots), dtype=float)
        compteur = {}

        # Comptage des occurrences
        for mot in mots:
            if mot in mot_to_index:
                compteur[mot] = compteur.get(mot, 0) + 1

        total = sum(compteur.values())  # Total de mots connus dans l'exemple

        for mot, count in compteur.items():
            v[mot_to_index[mot]] = count / total  # Fréquence relative

        res.append(v)

    return np.array(res)

#######################################################################################

def df2array_tfidf(df, index_mots):
    mot_to_index = {mot: i for i, mot in enumerate(index_mots)}
    N = len(df)

    # 1. Calculer les df (document frequency) pour chaque mot
    df_counts = np.zeros(len(index_mots))
    for mots in df["les_mots"]:
        mots_uniques = set(mots)
        for mot in mots_uniques:
            if mot in mot_to_index:
                df_counts[mot_to_index[mot]] += 1

    # 2. Calculer les idf
    idf = np.log(N / (1 + df_counts))

    # 3. Construction des vecteurs TF-IDF
    res = []
    for mots in df["les_mots"]:
        v = np.zeros(len(index_mots), dtype=float)
        compteur = {}

        for mot in mots:
            if mot in mot_to_index:
                compteur[mot] = compteur.get(mot, 0) + 1

        total = sum(compteur.values())  # total de mots dans le document
        for mot, count in compteur.items():
            idx = mot_to_index[mot]
            tf = count / total
            v[idx] = tf * idf[idx]

        res.append(v)

    return np.array(res)

#######################################################################################



#######################################################################################



#######################################################################################




