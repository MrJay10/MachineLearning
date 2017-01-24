# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:33:57 2016

@author: jay10
"""


import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # size of each data element in dataset
    
    # convert input element in the shape of dataset to find difference between input element and each data element
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  
    
    # Squared distance pairs
    sqDiffMat = diffMat**2
    
    # Sum of squared distance
    sqdistances = sqDiffMat.sum(axis=1)
    
    # actual distance
    distances = sqdistances ** 0.5
    
    # get sorting indices of the distances 
    sortedDistance = distances.argsort()
    
    classCount = {}
    
    for i in range(k):
        voteIlabel = labels[sortedDistance[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    # sort the nearest classes in reverse 
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]   # This is the class our input element belongs to.


def file2matrix(filename):
    fr = open(filename)
    numberOflines = len(fr.readlines())
    
    returnMat = np.zeros((numberOflines, 3))
    classLabelVector = []
    
    fr = open(filename)
    
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromline = line.split('\t')
        returnMat[index, :] = listFromline[:3]
        classLabelVector.append(listFromline[-1])
        index += 1
    
    return returnMat, classLabelVector


