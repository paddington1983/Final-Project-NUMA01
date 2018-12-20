# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:41:13 2018

@author: Ramon
"""
import unittest
from fractals import fractal2D
import numpy as np
import numpy.testing as nt

def function1(x):
    return x[0]**3 - 3*x[0]*x[1]**2 - 1

def dFunction1dx1(x):
    # derivative of function 1 wrt x1
    return 3*x[0]**2 - 3*x[1]**2

def dFunction1dx2(x):
    return 6*x[0]*x[1]

def function2(x):
    return 3*x[1]*x[0]**2 - x[1]**3

def dFunction2dx1(x):
    return 6*x[0]*x[1]

def dFunction2dx2(x):
    return 3*x[0]**2 - 3*x[1]**2

functionVector = np.array([function1, function2])
derivativeMatrix = np.matrix([[dFunction1dx1, dFunction1dx2], [dFunction2dx1, dFunction2dx2]])
fractal = fractal2D(functionVector, derivativeMatrix)

class FractalsTest(unittest.TestCase):
    
    # TESTS FOR INIT
    def test_init_with_functions_and_derivatives(self):
        # correct type of both arguments and correct number of derivatives
        
        output = fractal2D(functionVector, derivativeMatrix)
        
        # the arguments we put in are now stored in the object
        self.assertEqual(id(output.functionVector), id(functionVector))
        self.assertEqual(id(output.derivativeMatrix), id(derivativeMatrix))
        
        # and it contains an empty list of zeroes
        self.assertIsInstance(output.zeroes, list)
        self.assertFalse(output.zeroes)
    def test_init_with_functions_and_no_derivatives(self):
        # correct type of only argument
        
        output = fractal2D(functionVector)
        
        self.assertEqual(id(output.functionVector), id(functionVector))
        self.assertIsNone(output.derivativeMatrix)
        
        self.assertIsInstance(output.zeroes, list)
        self.assertFalse(output.zeroes)
    def test_init_with_functions_and_wrong_number_of_derivatives(self):
        # correct type of both arguments and wrong number of derivatives (if n is number or functions then derivative matrix should be n by n)
        
        # matrix is square, but with wrong n
        wrongDerivativeMatrix = np.matrix([[dFunction1dx1]])
        self.assertRaises(ValueError, fractal2D, functionVector, wrongDerivativeMatrix)
        
        # non-square matrix and wrong n in one dimension
        wrongDerivativeMatrix2 = np.matrix([[dFunction1dx1, dFunction1dx2, dFunction1dx2], [dFunction2dx1, dFunction2dx2, dFunction2dx2]])
        self.assertRaises(ValueError, fractal2D, functionVector, wrongDerivativeMatrix2)
    def test_init_with_no_functions_or_derivatives(self):
        # no arguments
        self.assertRaises(TypeError, fractal2D)
    def test_init_with_wrong_type_functions_and_no_derivatives(self):
        # wrong type of only argument
        self.assertRaises(TypeError, fractal2D, "this is not a function vector")
        self.assertRaises(TypeError, fractal2D, np.array(["this is not a function", "this is not a function either"]))
    def test_init_with_functions_and_wrong_type_derivatives(self):
        # correct type of first argument and wrong type of second argument
        self.assertRaises(TypeError, fractal2D, functionVector, "this is not a derivative matrix")
        self.assertRaises(TypeError, fractal2D, functionVector, np.matrix([["this is not a function", "this is not a function either"], ["this is not a function", "this is not a function either"]]))
    
    # TESTS FOR FINDZEROPOSITION
    def test_findZeroPosition_converges_to_zero1(self):
        # find the zero at 1, 0 in 0 iterations (very lucky guess)
        
        guess = np.array([1, 0])
        position, iterations = fractal.findZeroPosition(guess)
        
        # should return exactly the same vector and 0 iterations
        self.assertEqual(id(position), id(guess))
        self.assertEqual(iterations, 0)
    def test_findZeroPosition_converges_to_zero2(self):
        # find the zero at 10^(-1/3), -3^(1/2) * 10^(-1/3)
        
        guess = np.array([0.5, -1])
        position, _ = fractal.findZeroPosition(guess)
        
        nt.assert_allclose(position, np.array([10**(-1/3), -3**(1/2) * 10**(-1/3)]))
    def test_findZeroPosition_converges_to_zero3(self):
        # find the zero at 10^(-1/3), 3^(1/2) * 10^(-1/3)
        
        guess = np.array([0.5, 1])
        position, _ = fractal.findZeroPosition(guess)
        
        nt.assert_allclose(position, np.array([10**(-1/3), 3**(1/2) * 10**(-1/3)]))
        
        
suite = unittest.TestLoader().loadTestsFromTestCase(FractalsTest)
unittest.TextTestRunner(verbosity=2).run(suite)
