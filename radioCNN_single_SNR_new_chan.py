import tensorflow as tf
from matlab_params import *
import scipy.io as scio
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from radioCNN_single_SNR import RadioCNN, initiate_RadioCNN

# For reproductivity
np.random.seed(13)
tf.set_random_seed(2)

MODEL_PATH_NAME = './model/model_single_SNR.ckpt'

MAT_FILE_NAME = './conv_chan_data_AUG_idSNR_5.mat'
MAT_FILE_NEW_CHAN_NAME = './conv_chan_data_AUG_idSNR_5_new_chan.mat'

# Define main function
def main(argv=None):

    # Initiate & restpre RadioNN
    radioCNN_inst = initiate_RadioCNN(
        transfer_function = tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer,
        learning_rate=1e-4,
        weight_loss=0.05
    )
    radioCNN_inst.restore_model(MODEL_PATH_NAME)

    start_time = time.time()

    chan_data_old = scio.loadmat(MAT_FILE_NAME)
    chan_data_new = scio.loadmat(MAT_FILE_NEW_CHAN_NAME)

    Accuracy_old = radioCNN_inst.eval_accuracy(
        chan_data_old['test_data'],
        chan_data_old['test_tag']
    )
    SER_val_old = 1 - Accuracy_old

    Accuracy_new = radioCNN_inst.eval_accuracy(
        chan_data_new['test_data'],
        chan_data_new['test_tag']
    )
    SER_val_new = 1 - Accuracy_new

    print('Old SER = {0}, new SER = {1}'.format(SER_val_old, SER_val_new))
    print('Time consumed is {0} seconds'.format(time.time() - start_time))
    

if __name__ == '__main__':
    tf.app.run()