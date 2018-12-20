# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:40:25 2018

@author: Ramon
"""
import matplotlib.pyplot as plt
import numpy as np

print('hello, world!')

class fractal2D:
	def __init__(self, function,derivativematrix):
		self.function=function
		self.derivativematrix=derivativematrix
		self.zeroes=[]
