import tensorflow as tf
from matlab_params import *
import scipy.io as scio
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# Define network parameters
# Input layer
INPUT_DIM = [2, matlab_input_dim[1]]
INPUT_FLAT_DIM = INPUT_DIM[0] * INPUT_DIM[1]

# Conv layer 1
CONV1_INPUT_DIM = [2, INPUT_DIM[1], 1]
CONV1_KERNAL_DIM = [2, 20, 1, 8]
CONV1_KERNAL_STRIDES = [1, 1, 1, 1]
CONV1_OUTPUT_DIM = [2, INPUT_DIM[1], CONV1_KERNAL_DIM[3]]

# FC layer 1
FC1_INTPUT_DIM = CONV1_OUTPUT_DIM[0] * CONV1_OUTPUT_DIM[1] * CONV1_OUTPUT_DIM[2]
FC1_OUTPUT_DIM = 256

# FC layer 2
FC2_INPUT_DIM = FC1_OUTPUT_DIM
FC2_OUTPUT_DIM = 128

# OUTPUT layer
OUTPUT_DIM = matlab_output_dim

# Define functions to create variables
def weight_variable(shape, wl=0.0):
    initializer = tf.truncated_normal(shape=shape, stddev=0.1)
    var = tf.Variable(initializer)
    tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(var), wl))
    return var

def bias_variable(shape):
    initializer = tf.constant(0.1, shape=shape)
    return tf.Variable(initializer)

# Define network class
class RadioCNN(object):
    def __init__(self, transfer_function = tf.nn.relu, optimizer = tf.train.AdamOptimizer, learning_rate = 1e-4, weight_loss = 0.05):
        # initialize basics
        self.transfer = transfer_function
        self.weight_loss = weight_loss

        # initialize weights
        w_dict = self._initialize_weights()
        self.w_dict = w_dict

        # define learning rate
        self.learning_rate = learning_rate
        
        # define network structure
        self.x = tf.placeholder(
            dtype=tf.float32,
            shape=[None, INPUT_FLAT_DIM],
        )
        self.y_ = tf.placeholder(
            dtype=tf.float32,
            shape=[None, OUTPUT_DIM]
        )

        self.x_im = tf.reshape(self.x, shape = [-1, 2, INPUT_DIM[1], 1])
        
        # conv layer 1
        self.conv1_o = self.transfer(
            tf.nn.conv2d(self.x_im, w_dict['conv1_kernal'], strides=CONV1_KERNAL_STRIDES, padding='SAME') + w_dict['conv1_biases']
        )
        self.conv1_flat = tf.reshape(self.conv1_o, [-1, FC1_INTPUT_DIM])

        # FC layer 1
        self.fc1_o = self.transfer(
            tf.matmul(self.conv1_flat, w_dict['fc1_weights']) + w_dict['fc1_biases']
        )

        # FC layer 2
        self.fc2_o = self.transfer(
            tf.matmul(self.fc1_o, w_dict['fc2_weights']) + w_dict['fc2_biases']
        )

        # output layer
        self.y_conv = tf.nn.softmax(
            tf.matmul(self.fc2_o, w_dict['out_weights']) + w_dict['out_biases']
        )

        # define loss & optimizer
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), axis=[1]))
        self.loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
        self.optimizer = optimizer(self.learning_rate).minimize(self.loss)

        # define evaluation
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # self.sess
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        # finalize
        self.sess.graph.finalize()

    def _initialize_weights(self):
        w_dict = dict()

        w_dict['conv1_kernal'] = weight_variable(shape=CONV1_KERNAL_DIM, wl=0.0)
        w_dict['conv1_biases'] = bias_variable(shape=[CONV1_KERNAL_DIM[3]])

        w_dict['fc1_weights'] = weight_variable(shape=[FC1_INTPUT_DIM, FC1_OUTPUT_DIM], wl=self.weight_loss)
        w_dict['fc1_biases'] = bias_variable(shape=[FC1_OUTPUT_DIM])

        w_dict['fc2_weights'] = weight_variable(shape=[FC2_INPUT_DIM, FC2_OUTPUT_DIM], wl=self.weight_loss)
        w_dict['fc2_biases'] = bias_variable(shape=[FC2_OUTPUT_DIM])

        w_dict['out_weights'] = weight_variable(shape=[FC2_OUTPUT_DIM, OUTPUT_DIM], wl=0.0)
        w_dict['out_biases'] = bias_variable(shape=[OUTPUT_DIM])

        return w_dict
    
    def partial_train(self, X, Y):
        loss, _ = self.sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.x: X, self.y_: Y}
        )
        return loss

    def eval_accuracy(self, X, Y):
        accuracy = self.sess.run(
            self.accuracy,
            feed_dict={self.x: X, self.y_: Y}
        )
        return accuracy

    def get_weights(self):
        return self.w_dict

# Initiate RadioNN
def initiate_RadioCNN(transfer_function = tf.nn.softplus, optimizer=tf.train.AdamOptimizer, learning_rate=1e-4, weight_loss=0.05):
    radioCNN = RadioCNN(
        transfer_function = transfer_function,
        optimizer = optimizer, 
        learning_rate = learning_rate,
        weight_loss=weight_loss
    )
    return radioCNN

# Define training parameters
BATCH_SIZE = 100
LEARNING_RATE = 1e-4

def train(radioCNN, chan_data, output_process=False, show_performance=False, MAX_PILOT_NUM=50000, TRAINING_STEPS=10000):

    loss_rec = np.zeros(TRAINING_STEPS)
    temp_SER = np.zeros(int(TRAINING_STEPS / 1000))
    temp_SER_count = 0
    for id_step in range(TRAINING_STEPS):

        idx = np.random.randint(0, MAX_PILOT_NUM, size=BATCH_SIZE)
        X_batch = chan_data['sample_data'][idx, :]
        Y_batch = chan_data['sample_tag'][idx, :]

        if (id_step+1) % 1000 == 0 and output_process:
            temp_accuracy = radioCNN.eval_accuracy(
                X_batch,
                Y_batch
            )
            temp_SER[temp_SER_count] = 1 - temp_accuracy
            print('At step {0}, temporary SER is {1}.'.format(id_step, temp_SER[temp_SER_count]))
            
            temp_SER_count += 1
            
        loss_rec[id_step] = radioCNN.partial_train(
            X_batch,
            Y_batch
        )
    
    # Evaluate on the entire set of test data
    test_accuracy = radioCNN.eval_accuracy(
        chan_data['test_data'],
        chan_data['test_tag']
    )

    # Show total SER
    if show_performance:
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(int(TRAINING_STEPS / 1000)), temp_SER, 'bo-', label='SER')
        plt.legend()
        plt.yscale('log')

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(TRAINING_STEPS), loss_rec, 'b-', label='cross entropy')
        plt.legend()
        plt.show()

    return test_accuracy, radioCNN

# Define main function
SNR_NUM = 5
SNR_DB_RANGE = np.linspace(0, 7, SNR_NUM)

# SNR_NUM = 1
# SNR_DB_RANGE = [7]

def main(argv=None):

    # Initiate RadioNN
    radioCNN_inst = initiate_RadioCNN(
        transfer_function = tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer,
        learning_rate=1e-4,
        weight_loss=0.05
    )
    
    SER = np.zeros([SNR_NUM])
    for id_SNR in range(SNR_NUM):
        print('-------------SNR: {0} dB-------------'.format(SNR_DB_RANGE[id_SNR]))

        start_time = time.time()

        MAT_FILE_NAME = './conv_chan_data_AUG_idSNR_{0}'.format(id_SNR+1)
        chan_data = scio.loadmat(MAT_FILE_NAME)

        # Reference: MAX_PILOT_NUM = 1000, TRAINING_STEPS = 20000
        Accuracy, radioCNN_inst = train(
            radioCNN = radioCNN_inst,
            chan_data = chan_data, 
            output_process=True, 
            show_performance=False,
            MAX_PILOT_NUM=4000,
            TRAINING_STEPS=20000
        )
        SER[id_SNR] = 1 - Accuracy

        print('Total SER = {0}.'.format(SER[id_SNR]))
        print('Time consumed is {0} seconds'.format(time.time() - start_time))

    SER_benchmark = scio.loadmat('./SER_benchmark.mat')
    plt.plot(SNR_DB_RANGE, SER, 'bo-', label='RadioNN')
    plt.plot(SER_benchmark['SNRdBRng'][0, :], SER_benchmark['SER_mmse'][0, :], 'rx--', label='MMSE')
    plt.plot(SER_benchmark['SNRdBRng'][0, :], SER_benchmark['SER_ls'][0, :], 'r<-', label='ls')
    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate')
    plt.yscale('log')
    plt.show()

    scio.savemat('./RadioNN_performance.mat', {'SNR_DB_RANGE': SNR_DB_RANGE, 'SER': SER})


if __name__ == '__main__':
    tf.app.run()