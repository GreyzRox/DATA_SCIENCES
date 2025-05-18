# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2025

# Import de packages externes
import numpy as np
import pandas as pd
from iads import utils as ut
import copy
import graphviz as gv

from abc import ABC, abstractmethod


# ---------------------------

class Classifier(ABC):
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        
    @abstractmethod
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        pass

    @abstractmethod
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        pass

    @abstractmethod
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        pass

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        
        compteur = 0
        for desc,label in zip(desc_set,label_set):
            if (self.predict(desc) == label):
                compteur += 1
        return compteur/len(desc_set)

#######################################################################################

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)
        self.k = k
        self.data = None
        self.label = None

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.data = desc_set
        self.label = label_set
  
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        desc_copy = np.linalg.norm(self.data - x, axis = 1)
        sort = np.argsort(desc_copy)
        klabel = self.label[sort[:self.k]]
        p = (klabel==1).sum()/self.k
        return 2*(p-0.5)
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        score = self.score(x)
        if score > 0 :
            return 1
        else :
            return -1

#######################################################################################

class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)
        self.input_dimension=input_dimension
        self.v=np.random.uniform(-1,1,input_dimension)
        self.w=self.v/np.linalg.norm(self.v)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Aucun apprentissage nécessaires")
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score=self.score(x)
        return 1 if score>=0 else -1
        return 1 if score>=0 else -1

#######################################################################################

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self,input_dimension)
        self.lrate = learning_rate
        self.init = init
        if init:
            self.w = np.zeros(self.dimension)
        else:
            v = np.random.uniform(0,1,self.dimension)
            self.w = (2*v-1)*0.001
        self.allw =[self.w.copy()] # stockage des premiers poids
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        combined = list(zip(desc_set, label_set))
        np.random.shuffle(combined)
        shuffle_set,shuffle_label = zip(*combined)
        shuffle_set = np.array(shuffle_set)
        shuffle_label = np.array(shuffle_label)
        
        precedent_w = self.w.copy()
        
        for xi, yi in zip(shuffle_set, shuffle_label):
            yi_dot= np.dot(xi, self.w)  # Calcul du score
            yi_sign = np.sign(yi_dot)  # Prédiction
            if yi_sign != yi:
                self.w += self.lrate * yi * xi
                self.allw.append(self.w.copy())
            
        return np.linalg.norm(self.w-precedent_w)
        
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        nb_iterations = 0
        convergence = float('inf')
        convergence_list = []
        
        while nb_iterations<nb_max and convergence>seuil:
            convergence = self.train_step(desc_set,label_set)
            convergence_list.append(convergence)
            nb_iterations+=1
        return convergence_list
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x)>0:
            return 1
        else:
            return -1

    def get_allw(self):
        return self.allw

#######################################################################################


class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        combined = list(zip(desc_set, label_set))
        np.random.shuffle(combined)
        shuffle_set,shuffle_label = zip(*combined)
        shuffle_set = np.array(shuffle_set)
        shuffle_label = np.array(shuffle_label) 
        
        precedent_w = self.w.copy()
        
        for xi, yi in zip(shuffle_set, shuffle_label):
            if self.score(xi)*yi < 1:
                self.w += self.lrate*(yi - self.score(xi))*xi
                self.allw.append(self.w.copy())
            
        return np.linalg.norm(self.w-precedent_w)
        
        
#######################################################################################


class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """
        self.classifieur_binaire = cl_bin
        self.classifieurs = []
        self.classes = None
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.classes = np.unique(label_set)
        nCl = len(self.classes)

        for i in range(nCl):
            classif = copy.deepcopy(self.classifieur_binaire)

            ytmp = np.where(label_set == self.classes[i], 1, -1)
            classif.train(desc_set,ytmp)
            self.classifieurs.append(classif)
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.array([classifieur.score(x) for classifieur in self.classifieurs])
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        scores = self.score(x)
        classe_predite_index = np.argmax(scores)
        return self.classes[classe_predite_index]
        
#########################################################################

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    val,count = np.unique(Y, return_counts = True)
    arg = np.argmax(count)
    return val[arg]


#########################################################################

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    len_y = len(Y)
    liste_len_label = []
    df = pd.DataFrame({'label' : Y})
    groupe = df.groupby('label')
    res = [g['label'].tolist() for _, g in groupe]

    for e in res:
        liste_len_label.append(len(e))
    return shannon([i / len_y for i in liste_len_label])
            

########################################################################

def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """

    if len(P) > 1:
        log_k = np.log(len(P))
    else:
        log_k = 1

    return -(sum(p*np.log(p)/log_k for p in P if p > 0))
    
    
