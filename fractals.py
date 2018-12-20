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
    
    def plot(resolution, x1Start, x1Stop, x2Start, x2Stop):
        """Given two intervals one for the x1 variable and one for the x2 variable, this method
        will create a plot based on to which zero Newton's method converged. Each zero will have
        its own color in the image. 
        Arguments:
        Resolution = how many 'pixels' each row and column has in the final image. Higher is better but slower.
        x1Start, x1Stop = the interval of the first variable.
        x2Start, x2Stop = the interval of the second variable. """
        
        # Init two matricies both in resolution*resolution dimentions.
        # Columns will have the x1 entrys stored in columns.
        # columns: [[1 2 3 4]
        #          [1 2 3 4]
        #          [1 2 3 4]
        #          [1 2 3 4]]        
        # Rows will have the x2 entrys stored in rows.
        # rows: [[2 2 2 2]
        #        [3 3 3 3]
        #        [4 4 4 4]
        #        [5 5 5 5]]
        column = np.linspace(x1Start, x1Stop, resolution)
        row = np.linspace(x2Start, x2Stop, resolution)
        columns, rows = np.meshgrid(column, row)
        zerosMatrix = np.empty([resolution, resolution])
        
        # Build the zerosMAtrix. Each point in the grid (x1, x2) will be checked to wich
        # zero it will converge with Newtons method, this will be stored in zerosMatrix.
        for i, x1 in enumerate(row):
            for j, x2 in enumerate(column):
                zerosMatrix[i][j]= #TODO call to zeros function here args(x1, x2)
                # TODO handle if zeros function did not converge to a zero point.
        
        # Plot the resulting matrix with respect to the corodinates (x1, x2).
        plt.plot(rows, columns, zerosMatrix)
        # Make shure the hight and wiht of the plot are of equal length. 
        # So each colored rectangle is square like a pixel.
        plt.axis('scaled')
        plt.show()
        return
                
        
