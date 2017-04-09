import numpy as np
from sympy import *
#from sci import scipy as scipi


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

speed_of_light = 299792.458
estimate1 = np.array([0, 0, 6370, 0])
matr1 = np.array([15600,7540,20140, 0.07074]).astype(np.float64)
matr2 = np.array([18760,2750,18610, 0.07220]).astype(np.float64)
matr3 = np.array([17610,14630,13480, 0.07690]).astype(np.float64)
matr4 = np.array([19170,610,18390,0.07242]).astype(np.float64)
coordinates_matrix = Matrix([matr1, matr2, matr3, matr4])


# Task 1
def jacobimizer(estimate, coordinate_matrix):

    #  function_value_list = np.zeros(4).astype(np.float64)
    function_value_list = np.zeros(len(coordinate_matrix)).astype(np.float64)
    print("Function value list: {}".format(function_value_list))

    jacobi_matrix = np.zeros((len(coordinate_matrix), 4)).astype(np.float64)
    print("Jacobi matrix: {}".format(jacobi_matrix))

    print("Speed of light: {}".format(speed_of_light))

    #  squared_vals = np.zeros(4).astype(np.float64)
    squared_vals = np.zeros(len(coordinate_matrix)).astype(np.float64)
    print("Squared vals: {}".format(squared_vals))

    for i in range(0, len(coordinate_matrix)):
        print("Line 39: {}".format(i))
        temp_sqrt_list = np.zeros(4).astype(np.float64)
        for j in range(0, 3):
            temp_sqrt_list[j] = (np.power((estimate[j] - coordinate_matrix[i][j]), 2))
            print("temp_sqrt_list: {}".format(temp_sqrt_list))
        squared_vals[i] = (np.sqrt(sum(temp_sqrt_list)))
        print("Squared vals: {}".format(squared_vals))

    for i in range(len(coordinate_matrix)):
        function_value_list[i] = (squared_vals[i] + speed_of_light * (estimate[3] - coordinate_matrix[i][3]))
        print("function value list: {}".format(function_value_list))
    for i in range(len(coordinate_matrix)):
        for j in range(4):
            if j == 3:
                jacobi_matrix[i][j] = speed_of_light
            else:
                jacobi_matrix[i][j] = ((estimate[j] - coordinate_matrix[i][j]) / squared_vals[i])
            print("ij:{}{}, Jacobi:\n{}".format(i, j, jacobi_matrix))

    print("------END------")
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
phi1 = np.divide(np.pi, 9)
phi2 = np.divide(np.pi, 7)
phi3 = np.divide(np.pi, 5)
phi4 = np.divide(np.pi, 3)
phi5 = np.divide(np.pi, 6)
phi6 = np.divide(np.pi, 4)
phi7 = np.divide(np.pi, 2)
phi8 = 3*np.divide(np.pi, 10)
phi9 = 3*np.divide(np.pi, 11)
phi10 = 5*np.divide(np.pi, 11)
phi11 = 5*np.divide(np.pi, 12)

theta1 = np.divide(np.pi, 3)
theta2 = np.pi
theta3 = 3 * np.divide(np.pi, 2)
theta4 = 2 * np.pi
theta5 = np.divide(np.pi, 2)
theta6 = np.divide(np.pi, 5)
theta7 = np.divide(np.pi, 7)
theta8 = 4*np.divide(np.pi, 3)
theta9 = 5*np.divide(np.pi, 4)
theta10 = 6*np.divide(np.pi, 5)
theta11 = 7*np.divide(np.pi, 5)
rho = 26570

spc1 = [rho, phi1, theta1]
spc2 = [rho, phi2, theta2]
spc3 = [rho, phi3, theta3]
spc4 = [rho, phi4, theta4]
spc5 = [rho, phi5, theta5]
spc6 = [rho, phi6, theta6]
spc7 = [rho, phi7, theta7]
spc8 = [rho, phi8, theta8]
spc9 = [rho, phi9, theta9]
spc10 = [rho, phi10, theta10]
spc11 = [rho, phi11, theta11]

pc1 = [rho, phi2, theta3]
pc2 = [rho, phi1, theta2]
pc3 = [rho, phi4, theta1]
pc4 = [rho, phi3, theta6]
pc5 = [rho, phi6, theta5]
pc6 = [rho, phi7, theta4]
pc7 = [rho, phi8, theta9]
pc8 = [rho, phi7, theta8]
pc9 = [rho, phi10, theta7]
pc10 = [rho, phi9, theta11]
pc11 = [rho, phi11, theta10]


def task_2():
    c = coordinates_matrix
    Ux = zeros(3, 1)
    Uy = zeros(3, 1)
    Uz = zeros(3, 1)
    w = zeros(3, 1)
    D = zeros(3, 1)

    # (a-b) **2 = a ** 2 - 2ab + b ** 2
    # x1, y1 z1 & d1 subtracted in second quadratic sentence : 2(b)
    for i in range(0,3):
        Ux[i] = 2 * (c[i + 1, 0] - c[0, 0])
        Uy[i] = 2 * (c[i + 1, 1] - c[0, 1])
        Uz[i] = 2 * (c[i + 1, 2] - c[0, 2])
        D[i] = 2 * ((speed_of_light ** 2) * (c[0, 3] - c[i + 1, 3]))

    # w = b ** 2
    for i in range(0, 3):
        w[i] = c[0, 0] **2 - \
               c[i + 1, 0] **2 + c[0, 1] ** 2 - \
               c[i + 1, 1] **2 +c[0, 2] ** 2 - \
               c[i + 1, 2] **2 + (speed_of_light ** 2 * c[i + 1, 3] **2) - \
               (speed_of_light **2 * c[0, 3] ** 2)

    # z
    # for i in range(3):


    # solve for x in terms of D
    Xcol = np.array([Uy, Uz, Ux]).astype(np.float64)
    Dcol = np.array([Uy, Uz, D]).astype(np.float64)
    Wcol = np.array([Uy, Uz, w]).astype(np.float64)

    # determinant of x, d & w
    Xdeterminent = np.linalg.det(Xcol)
    Ddeterminent = np.linalg.det(Dcol)
    Wdeterminent = np.linalg.det(Wcol)



    # solve for y
    Ycol = np.array([Ux, Uz, Uy]).astype(np.float64)
    D2col = np.array([Ux, Uz, D]).astype(np.float64)
    W2col = np.array([Ux, Uz, w]).astype(np.float64)

    Ydeterminant = np.linalg.det(Ycol)
    D2determinant = np.linalg.det(D2col)
    W2determinant = np.linalg.det(W2col)


    # solve for z col
    Zcol = np.array([Ux, Uy, Uz]).astype(np.float64)
    D3col = np.array([Ux, Uy, D]).astype(np.float64)
    W3col = np.array([Ux, Uy, w]).astype(np.float64)

    zDeterminent = np.linalg.det(Zcol)
    D3Determinent = np.linalg.det(D3col)
    W3Determinent = np.linalg.det(W3col)


    # quadratic equation variables
    quadratic_a = (Ddeterminent/Xdeterminent) ** 2 + (D2determinant / Ydeterminant) ** 2 + (D3Determinent / zDeterminent) ** 2 - \
          speed_of_light ** 2
    quadratic_b = 2 * (Ddeterminent / Xdeterminent) * (Wdeterminent / Xdeterminent + c[0, 0]) + 2 * (D2determinant / Ydeterminant) * \
                                                                              (W2determinant / Ydeterminant + c[0,1]) + 2 * (D3Determinent / zDeterminent) * (W3Determinent / zDeterminent +c[0, 2]) + \
          2 * speed_of_light ** 2 * c[0, 3]
    quadratic_c = (Wdeterminent / Xdeterminent + c[0, 0]) ** 2 + (W2determinant / Ydeterminant + c[0, 1]) ** 2 + (W3Determinent / zDeterminent + c[0, 2]) ** 2 - speed_of_light **2 *c[0,3] ** 2


    if quadratic_b > 0:
        d1 = - ( quadratic_b + sqrt( quadratic_b ** 2 - 4 * quadratic_a * quadratic_c)) / (2 * quadratic_a)
        d2 = - ( 2 * quadratic_c) / ( quadratic_b + sqrt( quadratic_b ** 2 - 4 * quadratic_a * quadratic_c))
    else:
        d1 = - ( quadratic_b + sqrt( quadratic_b ** 2 - 4 * quadratic_a * quadratic_c)) / (2 * quadratic_a)
        d2 = - ( 2 * quadratic_c) / (- quadratic_b + sqrt( quadratic_b ** 2 - 4 * quadratic_a * quadratic_c))

    x1 = -(Wdeterminent / Xdeterminent + (Ddeterminent / Xdeterminent) * d1)

    x2 = -(Wdeterminent / Xdeterminent + (Ddeterminent / Xdeterminent) * d2)

    y1 = -(W2determinant / Ydeterminant + (D2determinant / Ydeterminant) * d1)
    y2 = -(W2determinant / Ydeterminant + (D2determinant / Ydeterminant) * d2)

    z1 = -(W3Determinent / zDeterminent + (D3Determinent / zDeterminent) * d1)
    z2 = -(W3Determinent / zDeterminent + (D3Determinent / zDeterminent) * d2)

    answ1 = np.array([x1, y1, z1, d1])
    answ2 = np.array([x2, y2, z2, d2])
    print('asnwer 1')
    print(answ1)
    print('answer 2')
    print(answ2)


def task_3():

    d = symbols('d')
    c = coordinates_matrix
    Ux = zeros(3, 1)
    Uy = zeros(3, 1)
    Uz = zeros(3, 1)
    w = zeros(3, 1)
    D = zeros(3, 1)

    for i in range(0,3):
        Ux[i] = 2 * c[i + 1, 0] - 2 * c[0, 0]
        Uy[i] = 2 * c[i + 1, 1] - 2 * c[0, 1]
        Uz[i] = 2 * c[i + 1, 2] - 2 * c[0, 2]
        D[i] = 2 * ((speed_of_light ** 2) * (c[0, 3] - c[i + 1, 3]))

        w[i] = c[0, 0] **2 -\
               c[i + 1, 0] **2 + c[0, 1] **2 -\
                c[i + 1, 1] **2 +c[0, 2] **2 -\
                c[i + 1, 2] **2 + (speed_of_light **2 * c[i + 1, 3] **2) -\
               (speed_of_light **2 * c[0,3] **2)

    # solve for x in terms of D
    Xcol = np.array([Uy, Uz, Ux]).astype(np.float64)
    Dcol = np.array([Uy, Uz, D]).astype(np.float64)
    Wcol = np.array([Uy, Uz, w]).astype(np.float64)

    # det of x, d & w
    Xdeterminent = np.linalg.det(Xcol)
    Ddeterminent = np.linalg.det(Dcol)
    Wdeterminent = np.linalg.det(Wcol)

    # solve for y
    Ycol = np.array([Ux, Uz, Uy]).astype(np.float64)
    D2col = np.array([Ux, Uz, D]).astype(np.float64)
    W2col = np.array([Ux, Uz, w]).astype(np.float64)

    Ydet = np.linalg.det(Ycol)
    D2det = np.linalg.det(D2col)
    W2det = np.linalg.det(W2col)

    # solve for z col
    Zcol = np.array([Ux, Uy, Uz]).astype(np.float64)
    D3col = np.array([Ux, Uy, D]).astype(np.float64)
    W3col = np.array([Ux, Uy, w]).astype(np.float64)

    zDeterminent = np.linalg.det(Zcol)
    D3Determinent = np.linalg.det(D3col)
    W3Determinent = np.linalg.det(W3col)

    # x, y, z = symbols('x y z')
    x = (-d * Ddeterminent - Wdeterminent) / Xdeterminent
    y = (-d * D2det - W2det) / Ydet
    z = (-d * D3Determinent - W3Determinent) / zDeterminent

    expression = ((x - c[0, 0]) ** 2 + (y - c[0, 1]) ** 2 + (z - c[0, 2]) ** 2 - (speed_of_light * (c[0, 3] - d)) ** 2)
    roots = solve(expression, d)

    root1 = roots[0]
    root2 = roots[1]

    x1 = x.subs(d, root1)
    y1 = y.subs(d, root1)
    z1 = z.subs(d, root1)

    x2 = x.subs(d, root2)
    y2 = y.subs(d, root2)
    z2 = z.subs(d, root2)

    print('x1 ' + str(x1))
    print('y1 ' + str(y1))
    print('z1 ' + str(z1))
    print('d1' + str(root1))

    print('x2 ' + str(x2))
    print('y2 ' + str(y2))
    print('z2 ' + str(z2))
    print('d2 ' + str(root2))


def calculate_coordinates2(spc_list):
    """

    :return:
    """

    coordinate_matrix = []

    for i in range(len(spc_list)):
        a = spc_list[i][0] * np.cos(spc_list[i][1]) * np.cos(spc_list[i][2])
        b = spc_list[i][0] * np.cos(spc_list[i][1]) * np.sin(spc_list[i][2])
        c = spc_list[i][0] * np.sin(spc_list[i][1])
        r = np.sqrt(pow(a, 2) + pow(b, 2) + pow((c - 6370), 2))
        t = 0.0001 + np.divide(r, 299792.458)
        coordinate_matrix.append([a, b, c, t])

    return np.array(coordinate_matrix).astype(np.float64)


def calculate_coordinates():
    """
    AKTIVITET 4

    :return:
    """
    # aspc1 = [26570, np.divide(np.pi, 9), np.divide(np.pi, 3)]
    # aspc2 = [26570, np.divide(np.pi, 7), np.pi]
    # aspc3 = [26570, np.divide(np.pi, 5), 3*(np.divide(np.pi, 2))]
    # aspc4 = [26570, np.divide(np.pi, 3), 2*np.pi]

    aspc_list = [spc1, spc2, spc3, spc4]
    coordinate_matrix = []

    for i in range(len(aspc_list)):
        a = aspc_list[i][0] * np.cos(aspc_list[i][1]) * np.cos(aspc_list[i][2])
        b = aspc_list[i][0] * np.cos(aspc_list[i][1]) * np.sin(aspc_list[i][2])
        c = aspc_list[i][0] * np.sin(aspc_list[i][1])
        r = np.sqrt(pow(a, 2) + pow(b, 2) + pow((c - 6370), 2))
        t = 0.0001 + np.divide(r, 299792.458)
        coordinate_matrix.append([a, b, c, t])

    return np.array(coordinate_matrix).astype(np.float64)

#  gir maksimal posisjonsfeil


def max_output_error(coordinate_matrix,  start_estimate, actual_xyz, e, newton_iterations):
    """
    Gir
    :param coordinate_matrix:
    :param start_estimate:
    :param actual_xyz:
    :param e:
    :param newton_iterations:
    :return:
    """

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
    """

    :param coordinate_matrix:
    :param e1:
    :param e2:
    :param e3:
    :param e4:
    :return:
    """

    coordinate_matrix[0, 3] += e1
    coordinate_matrix[1, 3] += e2
    coordinate_matrix[2, 3] += e3
    coordinate_matrix[3, 3] += e4

    return coordinate_matrix


def multi_newton(estimate, coordinate_matrix, number_of_iterations):
    """

    :param estimate:
    :param coordinate_matrix:
    :param number_of_iterations:
    :return:
    """

    return_list = jacobimizer(estimate, coordinate_matrix)
    # print(return_list[0])
    # print(return_list[1])
    for i in range(0, number_of_iterations):
        #  s = np.linalg.solve(return_list[1], return_list[0])  !!!!!!
        x = np.linalg.lstsq(return_list[1], return_list[0])
        estimate = np.subtract(estimate, x[0])
        return_list = jacobimizer(estimate, coordinate_matrix)
    return estimate


def the_satellite_question():
    #  sat_list = [spc1, spc2, spc3, spc4, spc5, spc6, spc7, spc8, spc9, spc10, spc11]
    sat_list = [pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10, pc11]

    eleven_sat_coordinates = calculate_coordinates2(sat_list)
    guess = [0, 0, 0, 0]
    actual_xyz = [0, 0, 6370]
    e = pow(10, -8)
    eleven_sat_output = max_output_error(eleven_sat_coordinates, guess, actual_xyz, e, 10)
    best_four_output = 10
    best_sat_nums = []
    best_sat_coordinates = []

    print("---Satellite coordinates---")
    for i in range(len(sat_list)):
        print("{}: {}".format(i, sat_list[i]))
    print("\n----OUTPUT----\n")

    for i in range(0, len(sat_list) - 3):
        for j in range(i+1, len(sat_list) - 2):
            for k in range(j+1, len(sat_list) - 1):
                for l in range(k+1, len(sat_list) - 0):
                    four_sat_list = [sat_list[i], sat_list[j], sat_list[k], sat_list[l]]
                    four_sat_coordinates = calculate_coordinates2(four_sat_list)
                    four_sat_output = max_output_error(four_sat_coordinates, guess, actual_xyz, e, 10)
                    print("----{},{},{},{}----".format(i, j, k, l))
                    print("output_error:{}\ndif:{}\n"
                          .format(four_sat_output, (four_sat_output - eleven_sat_output)))
                    if four_sat_output <= best_four_output:
                        best_four_output = four_sat_output
                        best_sat_nums = [i, j, k, l]
                        best_sat_coordinates = four_sat_coordinates

    print("Eleven sat output:{}".format(eleven_sat_output))
    print("Best four sat output:{}".format(best_four_output))
    print("Best four sat - eleven sat: = {}{}".format((best_four_output - eleven_sat_output), " (negative = better)"))
    print("Satellite numbers: {}".format(best_sat_nums))
    print("Satellite coordinates: {}".format(best_sat_coordinates))


def tight_satellites(err):
    phi = np.divide(np.pi, 4)
    theta = np.pi

    phi_list = [phi, (phi + np.multiply(phi, 0.025)), (phi + np.multiply(phi, -0.025)), (phi + np.multiply(phi, 0.02))]
    theta_list = [
        theta,
        (theta + np.multiply(theta1, 0.01)),
        (theta + np.multiply(theta, 0.02)),
        (theta + np.multiply(theta, -0.015))
    ]
    sat_list = []

    for i in range(4):
        sat_list.append([rho, phi_list[i], theta_list[i]])

    coordinates = calculate_coordinates2(sat_list)
    return max_output_error(coordinates, [0, 0, 0, 0], [0, 0, 6370], err, 10)

# e = pow(10, -8)
# input_error = e * 299792.458
# output_with_error = tight_satellites(e)
# emf_with_error = np.divide(output_with_error, input_error)
# print("Maximum position error (with e) = {}".format(output_with_error))
# print("EMF (with e) = {}".format(emf_with_error))
# print("")
#
# output_no_error = tight_satellites(0)
# emf_no_error = np.divide(output_no_error, input_error)
# print("Maximum position error (no e) = {}".format(output_no_error))
# print("EMF (no e) = {}".format(emf_no_error))




# # phi2 = phi1 + np.multiply(phi1, 0.025)
# # phi3 = phi1 - np.multiply(phi1, 0.025)
# # phi4 = phi1 + np.multiply(phi1, 0.04)
# #
# # theta1 = np.pi
# # theta2 = theta1 + np.multiply(theta1, 0.01)
# # theta3 = theta1 + np.multiply(theta1, 0.03)
# # theta4 = theta1 - np.multiply(theta1, 0.05)

# the_satellite_question()


# print("{}km / {}km = {}".format(output_error, input_error, result))
spc_list = [spc1, spc2, spc3, spc4]
akt1_list = [[15600, 7540, 20140, 0.07074],
             [18760, 2750, 18610, 0.07220],
             [17610, 14630, 13480, 0.07690],
             [19170, 610, 18390, 0.07242]]
akt1_result = multi_newton([0, 0, 6370, 0], akt1_list, 10)
print(akt1_result)

# coordinates = calculate_coordinates2(spc_list)
# output_error = max_output_error(coordinates, [0, 0, 0, 0], [0, 0, 6370], e, 10)
# input_error = e * 299792.458
# result = np.divide(output_error, input_error)
# print("{}km / {}km = {}".format(output_error, input_error, result))
# print(calculate_coordinates2(spc_list))

print('\noppgave 2')
task_2()
print('\noppgave 3')
task_3()