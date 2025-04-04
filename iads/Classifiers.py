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
import copy 


# ---------------------------

class Classifier:
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
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        pass
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        pass
    
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
        return compteur/len(label_set)

#######################################################################################

class ClassifierKNN(Classifier):
    def __init__(self, input_dimension, k):
        super().__init__(input_dimension)
        self.k = k
        self.desc = None
        self.label = None

    def train(self, desc_set, label_set):
        self.desc = desc_set
        self.label = label_set

    def score(self, x):
        if self.desc is None or self.label is None:
            raise ValueError("Classifieur non entraîné")
        
        distances = np.linalg.norm(self.desc - x, axis=1)
        indices = np.argsort(distances)[:self.k]
        k_labels = self.label[indices]
        return np.sum(k_labels == 1) / self.k

    def predict(self, x):
        return 1 if self.score(x) >= 0.5 else -1


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
            yi_dot= self.score(xi) # Calcul du score
            if yi_dot < 1:
                self.w += self.lrate*(yi - yi_dot)*xi
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
        self.cl_bin = cl_bin
        self.classifiers = []        
        
    def train(self, desc_set, label_set):
        self.classes = np.unique(label_set)
        self.classifiers = []

        for c in self.classes:
            # Création d'un classifieur indépendant pour chaque classe
            clf = copy.deepcopy(self.cl_bin)

            # Construction des labels binaires : +1 pour la classe c, -1 sinon
            bin_labels = np.array([1 if y == c else -1 for y in label_set])

            # Entraînement du classifieur binaire
            clf.train(desc_set, bin_labels)

            self.classifiers.append(clf)

       
    
    def score(self, x):
        scores = []
        for i, clf in enumerate(self.classifiers):
            s = clf.score(x)
            scores.append(s)
        return scores
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        scores = self.score(x)
        return self.classes[np.argmax(scores)]
        


