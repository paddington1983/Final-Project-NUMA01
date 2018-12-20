# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:41:13 2018

@author: Ramon
"""
import unittest
from fractals import fractal2D
import numpy as np

def function1(x1, x2):
    return x1**3 - 3*x1*x2**2 - 1

def dFunction1dx1(x1, x2):
    # derivative of function 1 wrt x1
    return 3*x1**2 - 3*x2**2

def dFunction1dx2(x1, x2):
    return 6*x1*x2

def function2(x1, x2):
    return 3*x2*x1**2 - x2**3

def dFunction2dx1(x1, x2):
    return 6*x1*x2

def dFunction2dx2(x1, x2):
    return 3*x1**2 - 3*x2**2

functionVector = np.array([function1, function2])
derivativeMatrix = np.matrix([[dFunction1dx1, dFunction1dx2], [dFunction2dx1, dFunction2dx2]])

class FractalsTest(unittest.TestCase):
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
    def test_init_with_functions_and_wrong_type_derivatives(self):
        # correct type of first argument and wrong type of second argument
        self.assertRaises(TypeError, fractal2D, functionVector, "this is not a derivative matrix")
        
suite = unittest.TestLoader().loadTestsFromTestCase(FractalsTest)
unittest.TextTestRunner(verbosity=2).run(suite)
