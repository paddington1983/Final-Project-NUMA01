# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:40:25 2018

@author: Ramon
"""
import matplotlib.pyplot as plt
import numpy as np

class fractal2D:
    def __init__(self, functionVector, derivativeMatrix):
        
        # check if functionVector is a vector of 2 functions
        if not isinstance(functionVector, np.ndarray):
            raise TypeError("functionVector must be of type array.")
        
        if functionVector.shape != (2,):
            raise ValueError("functionVector must be 2x1.")
        
        for function in functionVector:
            if not callable(function):
                raise TypeError("functionVector should consist of functions.")
        
        
        # check if derivativeMatrix is a 2x2 matrix of functions
        if not isinstance(derivativeMatrix, np.matrix):
            raise TypeError("derivativeMatrix should be of type matrix.")

        if  derivativeMatrix.shape != (2,2):
            raise ValueError("derivativeMatrix should be 2x2.")

        for row in derivativeMatrix:
            for function in row.A1:
                if not callable(function):
                    raise TypeError("derivativeMatrix should consist of functions.")


        # initialize the class
        self.functionVector = functionVector
        self.derivativeMatrix = derivativeMatrix
        self.zeroes = []


    def findZeroPosition(self, X):
        """Finds the position of a zero using Newton's Method, starting from the guessed X position (vector), also returns how many iterations it took.
        NOTE: does not validate its arguments as it is not meant for external use"""
        
        # the maximum number of iterations before giving up and concluding it will not converge to zero
        maxIterations = 50
        
        # a vector of zeroes we can easily compare with (and only have to create once)
        zeroVector = np.zeros(2)
        
        iterationsNeeded = 0
        
        while True:
            # calculate the vector of Y values using the vector of X values (the X position)
            Y = np.array([function(X) for function in self.functionVector])
            
            #print("X:", X, "Y:", Y)
            # we found the zero! exit the loop
            if np.allclose(Y, zeroVector):
                break
            
            # we give up, the max number of iterations has been reached and still no zero was found, return None as X position
            if iterationsNeeded == maxIterations:
                X = None
                break
            
            # we have not found the zero yet, apparently we need another iteration
            iterationsNeeded += 1
            
            # calculate the new X position:
            # first we need to calculate the values of all derivatives
            # use row.A1 as row is a matrix and A1 turns it into a vector, this makes it possible to loop over the values of the vector in the inner loop
            # looping over a matrix just generates more matrices
            J = np.matrix([[derivative(X) for derivative in row.A1] for row in self.derivativeMatrix])
            # then we can calculate the new X using the formula for Newton's method for several dimensions
            # we need the transpose of Y as matrix-vector multiplication needs to be done with a column vector
            # then we need to use A1 again to turn the result of the matrix-vector multiplication (which numpy thinks is a matrix) into a vector again
            X = X - (np.linalg.inv(J) * np.matrix(Y).T).A1
        
        return X, iterationsNeeded
    
   
    def findZeroIndex(self, x0):
        indexOfTheZero = -1
        position, iterationsNeeded = self.findZeroPosition(x0)
            
        #when we use 'findZeroPosition' method, there is no zero was found, then return None as x0 position and raise an error          
        #otherwise, if we can find a zero, then we use a loop to compare the newly found value with the already found zeroes(which have stored in the zeros list) with the tolerance'1.e-05'  
        #after comparing, we add the newly found value into the zeros list  
        if isinstance(position, np.ndarray):
            if len(self.zeroes)>0:                     #when the zeros list already has at least one value, then compare them.
                zeroWasFoundInList = False
                
                for i in range(len(self.zeroes)):
                    if np.allclose(self.zeroes[i], position):
                        zeroWasFoundInList = True
                        indexOfTheZero = i
                        break
                
                if not zeroWasFoundInList:
                    self.zeroes.append(position)
                    indexOfTheZero = len(self.zeroes) - 1

            else:                                 #if the zeros list is empty now , just add x0 into the zeros list 
                self.zeroes.append(position)
                indexOfTheZero = 0
            
        return (indexOfTheZero, iterationsNeeded)
        
        
    def plot(self, resolution, x1Start, x1Stop, x2Start, x2Stop):
        """Given two intervals one for the x1 variable and one for the x2 variable, this method
        will create a plot based on to which zero Newton's method converged. Each zero will have
        its own color in the image. 
        Arguments:
        Resolution = how many 'pixels' each row and column has in the final image. Higher is better but slower.
        x1Start, x1Stop = the interval of the first variable.
        x2Start, x2Stop = the interval of the second variable. """
        
        # Validate that the input is of correct data type.
        if not isinstance(resolution, int):
            raise TypeError('Resolution should be of type int.')
        if not all(isinstance(variable, (int or float)) for variable in [x1Start, x1Stop, x2Start, x2Stop]):
            raise TypeError('All start and stop variables should be of type int or fload.')
        
        # Validate that the intervals of x1 and x2 are valid.
        if x1Start >= x1Stop:
            raise ValueError('The x1 start value is higher than the x1 stop value.')
        if x2Start >= x2Stop:
            raise ValueError('The x2 start value is higher than the x2 stop value.')
        
        
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
                indexAndItterations = self.findZeroIndex(np.array([x1, x2]))
                zerosMatrix[i][j]= indexAndItterations[0]
                # TODO handle if zeros function did not converge to a zero point so if findzeroindex returns -1.
        
        # Plot the resulting matrix with respect to the corodinates (x1, x2).
        # (keep in mind that by plotting, the last column and row of zerosMatrix will be discarded.)
        plt.pcolor(rows, columns, zerosMatrix)
        # Make shure the hight and wiht of the plot are of equal length. 
        # So each colored rectangle is square like a pixel.
        plt.axis('scaled')
        plt.show()
        return
    
    

"""Task 4, first initialization of the fractal 2d class and first call to plot() to visualise the function stated in task 4."""
def function1(x):
    return x[0]**3 - 3*x[0]*x[1]**2 - 1

def function2(x):
    return 3*x[1]*x[0]**2 - x[1]**3

def dFunction1dx1(x):
    # derivative of function 1 wrt x1
    return 3*x[0]**2 - 3*x[1]**2

def dFunction1dx2(x):
    # derivative of function 1 wrt x2
    return -6*x[0]*x[1]

def dFunction2dx1(x):
    # derivative of function 2 wrt x1
    return 6*x[0]*x[1]

def dFunction2dx2(x):
    # derivative of function 2 wrt x2
    return 3*x[0]**2 - 3*x[1]**2

if __name__ == "__main__":
    task4FunctionVector = np.array([function1, function2])

    task4DerivativeMatrix = np.matrix([[dFunction1dx1, dFunction1dx2], [dFunction2dx1, dFunction2dx2]])

    firstFractal = fractal2D(task4FunctionVector, task4DerivativeMatrix)

    firstFractal.plot(40, -1, 1, -1, 1)                        
