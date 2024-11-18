
import numpy as np

def error(output_real, output_pred):

    output_pred = output_pred.clip(0,0.9999)
    return np.subtract(output_real, output_pred)

def J(error):

    return np.matmul(error, np.transpose(error))/np.shape(error)[1]

def d_error(error):

    return -2 * error / np.shape(error)[1]
