import numpy as np
import sympy as scipy
#from sci import scipy as scipi
print("hello world")
a = np.array([1,2,3])
print(a)

'''Solve the system (4.37) by using Multivariate Newtons Method. Find the receiver position
(x, y, z) near earth and time correction d for known, simultaneous satellite positions
(15600,7540,20140), (18760,2750,18610), (17610,14630,13480), (19170,610,18390)
in km, and measured time intervals 0.07074,0.07220,0.07690,0.07242 in seconds,
respectively. Set the initial vector to be (x0, y0, z0, d0) = (0,0,6370,0). As a check, the
answers are approximately (x, y, z) = (−41.77271,−16.78919,6370.0596), and
d = −3.201566 × 10−3 seconds.
'''

# i jacobi, skriv ut svar og kondisjonstall for siste matrise som blir regnet ut
# bruke newtons metode som verktoy i oppgave 7?
# ikke vits i aa sammenligne svar fra 5 og 6
# lønte det seg å ha 5-8 sattelitter

# Task 1
def fun1(estimate, coordinate_matrix):

    function_value_list = np.zeros(4).astype(np.float64)
    jacobi_matrix = np.zeros((len(coordinate_matrix),4)).astype(np.float64)

    speed_of_light = 299792.458

    squared_vals = np.zeros(4).astype(np.float64)
    for i in range(0, len(coordinate_matrix)):
        temp_sqrt_list = np.zeros(4).astype(np.float64)
        for j in range(0, 3):
            temp_sqrt_list[j] = (np.power((estimate[j] - coordinate_matrix[i][j]), 2))
        print(temp_sqrt_list)
        squared_vals[i] = (np.sqrt(sum(temp_sqrt_list)))

    for i in range(4):
        function_value_list[i] = (squared_vals[i] + speed_of_light * (estimate[3] - coordinate_matrix[i][3]))
    for i in range(len(coordinate_matrix)):
        temp_func_list = []
        for j in range(len(coordinate_matrix)):
            if j == 3:
                jacobi_matrix[i][j] = speed_of_light
            else:
                jacobi_matrix[i][j] = ((estimate[i] - coordinate_matrix[i][j]) / squared_vals[i])
    print(function_value_list)
    print('finish')

    return [function_value_list, jacobi_matrix]

estimate1 = np.array([0, 0, 6370, 0])
matr1 = np.array([15600,7540,20140, 0.07074]).astype(np.float64)
matr2 = np.array([18760,2750,18610, 0.07220]).astype(np.float64)
matr3 = np.array([17610,14630,13480, 0.07690]).astype(np.float64)
matr4 = np.array([19170,610,18390,0.07242]).astype(np.float64)
coordinates_matrix = np.array([matr1, matr2, matr3, matr4]).astype(np.float64)


def multi_newton(estimate, coordinate_matrix, number_of_iterations):
    return_list = fun1(estimate, coordinate_matrix)
    print(return_list[0])
    print(return_list[1])
    for i in range(0, number_of_iterations):
        s = np.linalg.solve(return_list[1], return_list[0])

        estimate = np.subtract(estimate, s)
        print(estimate)
        return_list = fun1(estimate, coordinate_matrix)
    return estimate


print(multi_newton(estimate1, coordinates_matrix, 10))