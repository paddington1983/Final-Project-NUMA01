# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:41:13 2018

@author: Ramon
"""
import unittest
import fractals as f
import numpy as np
import numpy.testing as nt


# FUNCTIONS FROM THE PROJECT
functionVector = np.array([f.function1, f.function2])
derivativeMatrix = np.matrix([[f.dFunction1dx1, f.dFunction1dx2], [f.dFunction2dx1, f.dFunction2dx2]])
fractal = f.fractal2D(functionVector, derivativeMatrix)

# FUNCTIONS FROM THE SLIDES
def f1(x):
    return 4*x[0]**2 - x[1]**3 + 28
def f2(x):
    return 3*x[0]**3 + 4*x[1]**2 - 145
def f1x(x):
    return 8*x[0]
def f1y(x):
    return -3*x[1]**2
def f2x(x):
    return 9*x[0]**2
def f2y(x):
    return 8*x[1]

class FractalsTest(unittest.TestCase):
    
    # TESTS FOR INIT
    def test_init_with_functions_and_derivatives(self):
        # correct type of both arguments and correct number of derivatives
        
        output = f.fractal2D(functionVector, derivativeMatrix)
        
        # the arguments we put in are now stored in the object
        self.assertEqual(id(output.functionVector), id(functionVector))
        self.assertEqual(id(output.derivativeMatrix), id(derivativeMatrix))
        
        # and it contains an empty list of zeroes
        self.assertIsInstance(output.zeroes, list)
        self.assertFalse(output.zeroes)
    def test_init_with_functions_and_no_derivatives(self):
        # correct type of only argument
        
        output = f.fractal2D(functionVector)
        
        self.assertEqual(id(output.functionVector), id(functionVector))
        self.assertIsInstance(output.derivativeMatrix, np.matrix)
        
        self.assertIsInstance(output.zeroes, list)
        self.assertFalse(output.zeroes)
    def test_init_with_functions_and_wrong_number_of_derivatives(self):
        # correct type of both arguments and wrong number of derivatives (if n is number or functions then derivative matrix should be n by n)
        
        # matrix is square, but with wrong n
        wrongDerivativeMatrix = np.matrix([[f.dFunction1dx1]])
        self.assertRaises(ValueError, f.fractal2D, functionVector, wrongDerivativeMatrix)
        
        # non-square matrix and wrong n in one dimension
        wrongDerivativeMatrix2 = np.matrix([[f.dFunction1dx1, f.dFunction1dx2, f.dFunction1dx2], [f.dFunction2dx1, f.dFunction2dx2, f.dFunction2dx2]])
        self.assertRaises(ValueError, f.fractal2D, functionVector, wrongDerivativeMatrix2)
    def test_init_with_no_functions_or_derivatives(self):
        # no arguments
        self.assertRaises(TypeError, f.fractal2D)
    def test_init_with_wrong_type_functions_and_no_derivatives(self):
        # wrong type of only argument
        self.assertRaises(TypeError, f.fractal2D, "this is not a function vector")
        self.assertRaises(TypeError, f.fractal2D, np.array(["this is not a function", "this is not a function either"]))
    def test_init_with_functions_and_wrong_type_derivatives(self):
        # correct type of first argument and wrong type of second argument
        self.assertRaises(TypeError, f.fractal2D, functionVector, "this is not a derivative matrix")
        self.assertRaises(TypeError, f.fractal2D, functionVector, np.matrix([["this is not a function", "this is not a function either"], ["this is not a function", "this is not a function either"]]))
    
    
    # TESTS FOR FINDZEROPOSITION
    def test_findZeroPosition_converges_to_zero1_lucky(self):
        # find the zero at 1, 0 in 0 iterations (very lucky guess)
        
        guess = np.array([1, 0])
        position, iterationsNeeded = fractal.findZeroPosition(guess)
        
        # should return exactly the same vector and 0 iterations
        self.assertEqual(id(position), id(guess))
        self.assertEqual(iterationsNeeded, 0)
    def test_findZeroPosition_converges_to_zero2_lucky(self):
        # find the zero at -1/2, sqrt(3)/2 in 0 iterations (very lucky guess)
        
        guess = np.array([-1/2, np.sqrt(3)/2])
        position, iterationsNeeded = fractal.findZeroPosition(guess)
        
        # should return exactly the same vector and 0 iterations
        self.assertEqual(id(position), id(guess))
        self.assertEqual(iterationsNeeded, 0)
    def test_findZeroPosition_converges_to_zero3_lucky(self):
        # find the zero at -1/2, -sqrt(3)/2 in 0 iterations (very lucky guess)
        
        guess = np.array([-1/2, -np.sqrt(3)/2])
        position, iterationsNeeded = fractal.findZeroPosition(guess)
        
        # should return exactly the same vector and 0 iterations
        self.assertEqual(id(position), id(guess))
        self.assertEqual(iterationsNeeded, 0)
    def test_findZeroPosition_converges_to_zero123(self):
        # find the zero at 1, 0 in a few iterations
        
        guess = np.array([0.9, 0.1])
        position, _ = fractal.findZeroPosition(guess)
        
        nt.assert_allclose(position, np.array([1, 0]), rtol=1e-05, atol=1e-08)
        
        # find the zero at -1/2, sqrt(3)/2 in a few iterations
        
        guess = np.array([-0.4, 0.9])
        position, _ = fractal.findZeroPosition(guess)
        
        nt.assert_allclose(position, np.array([-1/2, np.sqrt(3)/2]), rtol=1e-05, atol=1e-08)

        # find the zero at -1/2, -sqrt(3)/2 in a few iterations
        
        guess = np.array([-0.4, -0.9])
        position, _ = fractal.findZeroPosition(guess)
        
        nt.assert_allclose(position, np.array([-1/2, -np.sqrt(3)/2]), rtol=1e-05, atol=1e-08)
    def test_findZeroPosition_slide_example_works(self):
        # check if the example used in the slides works, we know from that example where it converges to and in how many iterations
        
        slidesFractal = f.fractal2D(np.array([f1, f2]), np.matrix([[f1x, f1y], [f2x, f2y]]))
        position, iterationsNeeded = slidesFractal.findZeroPosition(np.array([1, 1]))
        
        nt.assert_allclose(position, np.array([3, 4]))
        self.assertEqual(iterationsNeeded, 10)
        
        
    # TESTS FOR FINDZEROINDEX
    def test_findZeroIndex_slide_example_works(self):
        # check if the example used in the slides works, we know from that example where it converges to and in how many iterations
        
        slidesFractal = f.fractal2D(np.array([f1, f2]), np.matrix([[f1x, f1y], [f2x, f2y]]))
        index, iterationsNeeded = slidesFractal.findZeroIndex(np.array([1, 1]))
        
        self.assertEqual(index, 0)
        self.assertEqual(iterationsNeeded, 10)
        self.assertEqual(len(slidesFractal.zeroes), 1)
        nt.assert_allclose(slidesFractal.zeroes[0], np.array([3, 4]))
        
        # calling the method a second time should not increase the length of the zeroes list (in fact, it should not change anything)
        index, iterationsNeeded = slidesFractal.findZeroIndex(np.array([1, 1]))
        self.assertEqual(index, 0)
        self.assertEqual(iterationsNeeded, 10)
        self.assertEqual(len(slidesFractal.zeroes), 1)
        nt.assert_allclose(slidesFractal.zeroes[0], np.array([3, 4]))
    
    
    # TESTS FOR PLOT
    def test_plot_with_correct_arguments(self):
        # we dont want to show the plot, so replace the pyplot functions with a function that does nothing, first back them up though
        show_backup = f.plt.show
        pcolor_backup = f.plt.pcolor
        axis_backup = f.plt.axis
        doNothing = lambda *args, **kwargs: None
        f.plt.show = doNothing
        f.plt.pcolor = doNothing
        f.plt.axis = doNothing
        
        # now we can test without plotting
        fractal.plot(2, -1, 1, -1, 1)
        
        # now restore these functions
        f.plt.show = show_backup
        f.plt.pcolor = pcolor_backup
        f.plt.axis = axis_backup
    def test_plot_with_wrong_resolution_type(self):
        self.assertRaises(TypeError, fractal.plot, "2", -1, 1, -1, 1)
    def test_plot_with_nonpositive_resolution(self):
        self.assertRaises(ValueError, fractal.plot, -2, -1, 1, -1, 1)
        self.assertRaises(ValueError, fractal.plot, 0, -1, 1, -1, 1)
    def test_plot_with_wrong_x_values_type(self):
        self.assertRaises(TypeError, fractal.plot, 2, "-1", 1, -1, 1)
        self.assertRaises(TypeError, fractal.plot, 2, -1, "1", -1, 1)
        self.assertRaises(TypeError, fractal.plot, 2, -1, 1, "-1", 1)
        self.assertRaises(TypeError, fractal.plot, 2, -1, 1, -1, "1")
    def test_plot_with_nonpositive_x1_interval(self):
        self.assertRaises(ValueError, fractal.plot, 2, 1, -1, -1, 1)
        self.assertRaises(ValueError, fractal.plot, 2, 1, 1, -1, 1)
    def test_plot_with_nonpositive_x2_interval(self):
        self.assertRaises(ValueError, fractal.plot, 2, -1, 1, 1, -1)
        self.assertRaises(ValueError, fractal.plot, 2, -1, 1, 1, 1)


    # TESTS FOR TASK 6
    def test_findZeroPosition_slide_example_works_no_derivative(self):
        slidesFractal = f.fractal2D(np.array([f1, f2]))
        position, _ = slidesFractal.findZeroPosition(np.array([1, 1]))
        
        nt.assert_allclose(position, np.array([3, 4]))
        

suite = unittest.TestLoader().loadTestsFromTestCase(FractalsTest)
unittest.TextTestRunner(verbosity=2).run(suite)
