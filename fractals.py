# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:40:25 2018

@author: Ramon
"""
import matplotlib.pyplot as plt
import numpy as np

class fractal2D:
    def __init__(self, functionVector, derivativeMatrix):
        
        """It is verry important to validate your variables before you start working with them. So I moved this block to the biginning of the 
        init method!"""
        
        """Here I made an example of how to apply isinstance in the right way. It was basically correct, but I added the 'np.matrix'
        Now it will chack if derivativeMatrix is of data type matrix. So when you use isinstance you first write the variable name you 
        want to check and than ater a comma you write the data type you want it to check."""
        if not isinstance(derivativeMatrix, np.matrix ):
            raise TypeError("Derivative matrix should be of type matrix")
        
        """Now you need to check if the derivativeMatrix has the correct dimentions.
        To do this use derivativeMatrix.shape == 'a 2x2 marix' 
        Here is a link to the information on the function shape and how to use it: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matrix.shape.html
        Keep in mind that .shape returns a tuple so you need to comape what it returns to a tuple. So think about what .shape should return and check if it indeed retuns this. 
        If it dosn't return what it is supposed to than rais a ValueError. That works the same as a TypeError but with a different name, aka ValueError."""
        
        """Now we need one more validation for the derivativeMatrix variable. We need to check if the stuf in the derivativeMartix has the correct type. 
        This means that we need to check if the entrys in the matrix are of a 'function' type. The derivatives are functions. To do this in python
        we need the method callable(). Here is a link to the usage of callable: https://www.programiz.com/python-programming/methods/built-in/callable
        In general if you give callable some argument it will check if python can execute what you give it as an argument. This is what we
        need to check if somethin is a function. 
        To check if the derivative matrix has four function we need to loop over the entire matrix and check if every cell in the marix 
        contains a funtion, aka if callable returns 'true'. If callable returns false we need to raise an TypeError to tell the user that they made
        a matrix that we can not use."""
        
        """Now we can make sure that functionVactor is of the correct type and shape. 
        First we need to check if functionVector is of type np.ndarray. Here I have compleated what you had written as an example. 
        Note the 'np.ndarray' and the slightly altered error message so users get more information on what they did wrong."""
        if not isinstance(functionVector, np.ndarray):
            raise TypeError("FunctionVector must be of type array.")
        """Now we need to check if functionVector has the correct shape. So we can use the shape method we used above again in this case as well.
        Now if shape is used on a one dimentional matrix. So just with one row, it will return a tuple of the form (length, 'nothing') so we need to addapt our check here accordingly.
        If the length if the array is not correct we need to raise a ValueError again but with a different message so the user knows what
        went wrong."""
        # if not isinstance(functionVector, ):
            #  raise TypeError("there must be two functions")
        
        """Finally we need to check if the entrys in the array are of type function again. So we can use callable again on every entry in the array. 
        If one of the entrys in the functionVector is not a function we need to raise a TypeError again and let the user know what he or she did wrong this time. 
        
        If you have all these checks working correctly you can see that if you run the test file (just open the test files tab and click run) than you chould see that there 
        are no more errors or fails. YEAY! ;-)
        I hope this makes it all a lot more clear, if you have any more questions please let me know.  I will do my best to help.
        """
        
        
        self.functionVector = functionVector
        self.derivativeMatrix = derivativeMatrix
        self.zeroes = []
        
        


    def findZeroPosition(self, X):
        """Finds the position of a zero using Newton's Method, starting from the guessed X position (vector), also returns how many iterations it took"""
        
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a vector (numpy array)")
        
        if X.size != 2:
            raise ValueError("X should be a vector of length 2")
        
        # the maximum number of iterations before giving up and concluding it will not converge to zero
        maxIterations = 25
        
        # a vector of zeroes we can easily compare with (and only have to create once)
        zeroVector = np.zeros(2)
        
        iterationsNeeded = 0
        
        while True:
            # calculate the vector of Y values using the vector of X values (the X position)
            Y = np.array([function(X) for function in self.functionVector])
            
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
    
   #TASK 3 
   
    def findZeroIndex(self, x0):
        if not isinstance(x0, np.ndarray):
            raise TypeError("x0 should be a vector (numpy array)")
            
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
            
   ######     
        
        
    def plot(resolution, x1Start, x1Stop, x2Start, x2Stop):
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
                zerosMatrix[i][j]= 0 #TODO call to zeros function here args(x1, x2)
                # TODO handle if zeros function did not converge to a zero point.
        
        # Plot the resulting matrix with respect to the corodinates (x1, x2).
        # (keep in mind that by plotting, the last column and row of zerosMatrix will be discarded.)
        plt.plot(rows, columns, zerosMatrix)
        # Make shure the hight and wiht of the plot are of equal length. 
        # So each colored rectangle is square like a pixel.
        plt.axis('scaled')
        plt.show()
        return
                
        
