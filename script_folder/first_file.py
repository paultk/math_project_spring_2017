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
    jacobi_matrix = np.zeros((len(coordinate_matrix), 4)).astype(np.float64)

    speed_of_light = 299792.458

    squared_vals = np.zeros(4).astype(np.float64)
    for i in range(0, len(coordinate_matrix)):
        temp_sqrt_list = np.zeros(4).astype(np.float64)
        for j in range(0, 3):
            temp_sqrt_list[j] = (np.power((estimate[j] - coordinate_matrix[i][j]), 2))
        #  print(temp_sqrt_list)
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
    #  print(function_value_list)
    #  print('finish')

    return [function_value_list, jacobi_matrix]

# t1 = 0.083876
# t2 = 0.081784
# t3 = 0.07816
# t4 = 0.07113
#
# estimate1 = np.array([0, 0, 6370, 0.0001])
# matr1 = np.array([12484, 21623, 9087, t1]).astype(np.float64)
# matr2 = np.array([-23939, 0, 11528, t2]).astype(np.float64)
# matr3 = np.array([0, -21496, 15617, t3]).astype(np.float64)
# matr4 = np.array([13285, 0, 23010, t4]).astype(np.float64)
# coordinates_matrix = np.array([matr1, matr2, matr3, matr4]).astype(np.float64)


def calculate_coordinates():
    spc1 = [26570, np.divide(np.pi, 9), np.divide(np.pi, 3)]
    spc2 = [26570, np.divide(np.pi, 7), np.pi]
    spc3 = [26570, np.divide(np.pi, 5), 3*(np.divide(np.pi, 2))]
    spc4 = [26570, np.divide(np.pi, 3), 2*np.pi]

    spc_list = [spc1, spc2, spc3, spc4]
    coordinate_matrix = []

    for i in range(len(spc_list)):
        a = spc_list[i][0] * np.cos(spc_list[i][1]) * np.cos(spc_list[i][2])
        b = spc_list[i][0] * np.cos(spc_list[i][1]) * np.sin(spc_list[i][2])
        c = spc_list[i][0] * np.sin(spc_list[i][1])
        r = np.sqrt(pow(a, 2) + pow(b, 2) + pow((c - 6370), 2))
        t = 0.0001 + np.divide(r, 299792.458)
        coordinate_matrix.append([a, b, c, t])

    return np.array(coordinate_matrix).astype(np.float64)


def max_output_error(coordinate_matrix,  start_estimate, actual_xyz, e, newton_iterations):
    error_list = []

    for i in range(-1, 3, 2):
        for j in range(-1, 3, 2):
            for k in range(-1, 3, 2):
                for l in range(-1, 3, 2):
                    cm_with_error = add_timing_error(coordinate_matrix, i*e, j*e, k*e, l*e)
                    mn_estimate = multi_newton(start_estimate, cm_with_error, newton_iterations)
                    calculated_xyz = [mn_estimate[0], mn_estimate[1], mn_estimate[2]]
                    error = np.linalg.norm((np.subtract(actual_xyz, calculated_xyz)), np.inf)
                    print("e:{}, i:{}, j:{}, k:{}, l:{}".format(e, i, j, k, l))
                    print("Calculated: {} \nError: {} \n".format(mn_estimate, error))
                    error_list.append(error)
    return np.linalg.norm(error_list, np.inf)


def add_timing_error(coordinate_matrix, e1, e2, e3, e4):
    coordinate_matrix[0, 3] += e1
    coordinate_matrix[1, 3] += e2
    coordinate_matrix[2, 3] += e3
    coordinate_matrix[3, 3] += e4

    return coordinate_matrix


def multi_newton(estimate, coordinate_matrix, number_of_iterations):
    return_list = fun1(estimate, coordinate_matrix)
    print(return_list[0])
    print(return_list[1])
    for i in range(0, number_of_iterations):
        s = np.linalg.solve(return_list[1], return_list[0])

        estimate = np.subtract(estimate, s)
        return_list = fun1(estimate, coordinate_matrix)
    return estimate

e = pow(10, -8)
coordinates = calculate_coordinates()
output_error = max_output_error(coordinates, [0, 0, 0, 0], [0, 0, 6370], e, 10)
input_error = e * 299792.458
result = np.divide(output_error, input_error)
print("{}km / {}km = {}".format(output_error, input_error, result))
print(calculate_coordinates())
# output_error = max_error(matr1, matr2, matr3, matr4, 10)
# input_error = pow(10, -8) * 299792458
# result = np.divide(output_error, input_error)
# print("{}m/{}m = {}".format(output_error, input_error, result))
