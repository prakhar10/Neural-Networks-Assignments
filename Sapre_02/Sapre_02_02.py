# Sapre, Prakhar
# 1001-514-586
# 2018-09-23
# Assignment-02-02

import numpy as np


#This method will calculate the actual value whic the selected activation function will provide as output
def calculate_activation_function(net_value,transfer_function_type):
    actual_value=0
    if transfer_function_type == "Symmetrical Hard Limit":
        if net_value < 0:
            actual_value = -1
        elif net_value >= 0:
            actual_value = 1
    elif transfer_function_type == "Hyperbolic Tangent":
        actual_value = np.tanh(net_value)
    elif transfer_function_type == "Linear":
        actual_value = net_value
    return actual_value


# This method will loop for 100 epochs and calculate the line equation to plot it on the graph
def train_perceptron(transfer_function_type,first_weight,second_weight,bias,input_points,target_value):
    for i in range(0,4):
        net_value = first_weight * input_points[i][0] + second_weight * input_points[i][1] + bias
        actual_value = calculate_activation_function(net_value,transfer_function_type)
        error = target_value[i] - actual_value
        first_weight += (error * input_points[i][0])
        second_weight += (error * input_points[i][1])
        bias += error
    return first_weight, second_weight, bias
