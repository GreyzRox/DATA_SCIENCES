# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy

# ------------------------ 


#######################################################################################

def normalisation(df):
    copie_df = df.copy()
    
    for col in copie_df.columns:
        maxi = copie_df[col].max()
        mini = copie_df[col].min()
        copie_df[col] = (copie_df[col] - mini)/(maxi-mini)

    return copie_df

#######################################################################################

def dist_euclidienne(v1,v2):
    return np.linalg.norm(v2-v1)

#######################################################################################

def centroide(v1):
    return np.mean(v1, axis = 0)

#######################################################################################

def dist_centroides(v1,v2):
    return dist_euclidienne(centroide(v1),centroide(v2))

#######################################################################################

def dist_linkage(linkage, a1, a2):
    res = cdist(a1,a2, 'euclidean')
    if linkage == 'complete':
        return np.max(res)
    if linkage == 'simple':
        return np.min(res)
    if linkage == 'average':
        return np.mean(res)

#######################################################################################

def initialise_CHA(df):
    partition = {}

    for index in df.index:
        partition[index] = [index]
    return partition

#######################################################################################

def fusionne(df,P0,verbose=False):
    dist_min = math.inf
    key1, key2 = 0, 0

    for c1 in P0 :
        for c2 in P0 :
            dist = dist_centroides(df.values[P0[c1]], df.values[P0[c2]])
            if c1!=c2 and dist < dist_min :
                dist_min = dist
                key1, key2 = c1, c2
    if verbose : print(f"fusionne: distance mininimale trouvée entre  [{key1}, {key2}]  =  {dist_min}") 
    if verbose : print(f"fusionne: les 2 clusters dont les clés sont  [{key1}, {key2}]  sont fusionnés") 
    
    P1 = P0.copy()
    new_key = max(P1)+1
    P1[new_key] = P1[key1] + P1[key2]
    if verbose : print(f"fusionne: on crée la  nouvelle clé {new_key}  dans le dictionnaire.") 
    P1.pop(key1)
    P1.pop(key2)
    if verbose : print(f"fusionne: les clés de  [{key1}, {key2}]  sont supprimées car leurs clusters ont été fusionnés.") 

    return (P1, key1, key2, dist_min)


#######################################################################################


def fusionne_linkage(df,linkage,P0,verbose=False):
    dist_min = math.inf
    key1, key2 = 0, 0

    for c1 in P0 :
        for c2 in P0 :
            dist = dist_linkage(linkage,df.values[P0[c1]], df.values[P0[c2]])
            if c1!=c2 and dist < dist_min :
                dist_min = dist
                key1, key2 = c1, c2
    if verbose : print(f"fusionne: distance mininimale trouvée entre  [{key1}, {key2}]  =  {dist_min}") 
    if verbose : print(f"fusionne: les 2 clusters dont les clés sont  [{key1}, {key2}]  sont fusionnés") 
    
    P1 = P0.copy()
    new_key = max(P1)+1
    P1[new_key] = P1[key1] + P1[key2]
    if verbose : print(f"fusionne: on crée la  nouvelle clé {new_key}  dans le dictionnaire.") 
    P1.pop(key1)
    P1.pop(key2)
    if verbose : print(f"fusionne: les clés de  [{key1}, {key2}]  sont supprimées car leurs clusters ont été fusionnés.") 

    return (P1, key1, key2, dist_min)

#######################################################################################

def CHA_centroid(df, verbose=False, dendrogramme=False):
    res = []
    partition = initialise_CHA(df)
    
    if verbose: print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")
    while len(partition)!=1:
        partition, key1, key2, dist = fusionne(df, partition, verbose)
        nb_examples = len(partition[max(partition)])
        if verbose:
            print(f"CHA_centroid: une fusion réalisée de  {key1}  avec  {key2} de distance  {dist:.4f}")
            print(f"CHA_centroid: le nouveau cluster contient  {nb_examples}  exemples")
        res.append([key1, key2, dist, nb_examples])
    if verbose: print("CHA_centroid: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendogramme (Approche Centroid Linkage)', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,
        )

        plt.show()
    
    return res

#######################################################################################

def CHA_complete(df, verbose=False,dendrogramme=False):
    res = []
    partition = initialise_CHA(df)
    
    if verbose: print("CHA_complete: clustering hiérarchique ascendant, version Complete Linkage")
    while len(partition)!=1:
        partition, key1, key2, dist = fusionne_linkage(df, 'complete', partition, verbose)
        nb_examples = len(partition[max(partition)])
        if verbose:
            print(f"CHA_complete: une fusion réalisée de  {key1}  avec  {key2} de distance  {dist:.4f}")
            print(f"CHA_complete: le nouveau cluster contient  {nb_examples}  exemples")
        res.append([key1, key2, dist, nb_examples])
    if verbose: print("CHA_complete: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme (Approche Complete Linkage)', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,
        )

        plt.show()
    
    return res

#######################################################################################

def CHA_simple(df, verbose=False,dendrogramme=False):
    res = []
    partition = initialise_CHA(df)
    
    if verbose: print("CHA_simple: clustering hiérarchique ascendant, version Simple Linkage")
    while len(partition)!=1:
        partition, key1, key2, dist = fusionne_linkage(df, 'simple', partition, verbose)
        nb_examples = len(partition[max(partition)])
        if verbose:
            print(f"CHA_complete: une fusion réalisée de  {key1}  avec  {key2} de distance  {dist:.4f}")
            print(f"CHA_complete: le nouveau cluster contient  {nb_examples}  exemples")
        res.append([key1, key2, dist, nb_examples])
    if verbose: print("CHA_simple: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme (Approche Simple Linkage)', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,
        )

        plt.show()
    
    return res

#######################################################################################

def CHA_average(df, verbose=False,dendrogramme=False):
    res = []
    partition = initialise_CHA(df)
    
    if verbose: print("CHA_average: clustering hiérarchique ascendant, version Average Linkage")
    while len(partition)!=1:
        partition, key1, key2, dist = fusionne_linkage(df, 'average', partition, verbose)
        nb_examples = len(partition[max(partition)])
        if verbose:
            print(f"CHA_complete: une fusion réalisée de  {key1}  avec  {key2} de distance  {dist:.4f}")
            print(f"CHA_complete: le nouveau cluster contient  {nb_examples}  exemples")
        res.append([key1, key2, dist, nb_examples])
    if verbose: print("CHA_average: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme (Approche Average Linkage)', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,
        )

        plt.show()
    
    return res

#######################################################################################

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    match linkage:
        case 'centroid':
            return CHA_centroid(DF,verbose,dendrogramme)
        case 'complete':
            return CHA_complete(DF,verbose,dendrogramme)
        case 'simple':
            return CHA_simple(DF,verbose,dendrogramme)
        case 'average':
            return CHA_average(DF,verbose,dendrogramme)

#######################################################################################



#######################################################################################








































