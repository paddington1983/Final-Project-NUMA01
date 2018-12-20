# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:40:25 2018

@author: Ramon
"""
import matplotlib.pyplot as plt
import numpy as np

class fractal2D:
    def __init__(self, functionVector, derivativeMatrix):
        self.functionVector = functionVector
        self.derivativeMatrix = derivativeMatrix
        self.zeroes = []
    def findZeroPosition(self, guess):
        """Finds the position of a zero using Newton's Method, starting from guess, also returns how many iterations it took"""
        
        return guess, 0
