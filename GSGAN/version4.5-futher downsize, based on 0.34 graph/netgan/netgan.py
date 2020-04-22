"""
Implementation of the method proposed in the paper:
'Adversarial Attacks on Classification Models for Graphs'
by Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, Stephan Günnemann
Published at ICML 2018 in Stockholm, Sweden.

Copyright (C) 2018
Daniel Zügner
Technical University of Munich
"""

import tensorflow as tf
from netgan import utils
from netgan.attention_layer import *
import scipy
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import community
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.monitor_interval = 0
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
import pandas as pd
import networkx as nx
import warnings
from scipy import sparse

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

class NetGAN:
    """
    NetGAN class, an implicit generative model for graphs using random walks.
    """

    def __init__(self, N, rw_len, 
                walk_generator_1=None, privacy_walk_generator_1=None, 
                walk_generator_2=None, privacy_walk_generator_2=None, 
                walk_generator_3=None, privacy_walk_generator_3=None, 
                generator_layers=[40], discriminator_layers=[30],
                W_down_generator_size=128, W_down_discriminator_size=128, batch_size=128, noise_dim=16,
                noise_type="Gaussian", learning_rate=0.0003, disc_iters=3, wasserstein_penalty=10,
                l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, temp_start=5.0, min_temperature=0.5,
                temperature_decay=1-5e-5, seed=15, gpu_id=0, use_gumbel=True, legacy_generator=False,
                dir_name="fuck"):
        """
        Initialize NetGAN.

        Parameters
        ----------
        N: int
           Number of nodes in the graph to generate.
        rw_len: int
                Length of random walks to generate.
        walk_generator: function
                        Function that generates a single random walk and takes no arguments.
        privacy_walk_generator: function (privacy garuntee)
                        Function that generates a single random walk and takes no arguments.
        generator_layers: list of integers, default: [40], i.e. a single layer with 40 units.
                          The layer sizes of the generator LSTM layers
        discriminator_layers: list of integers, default: [30], i.e. a single layer with 30 units.
                              The sizes of the discriminator LSTM layers
        W_down_generator_size: int, default: 128
                               The size of the weight matrix W_down of the generator. See our paper for details.
        W_down_discriminator_size: int, default: 128
                                   The size of the weight matrix W_down of the discriminator. See our paper for details.
        batch_size: int, default: 128
                    The batch size.
        noise_dim: int, default: 16
                   The dimension of the random noise that is used as input to the generator.
        noise_type: str in ["Gaussian", "Uniform], default: "Gaussian"
                    The noise type to feed into the generator.
        learning_rate: float, default: 0.0003
                       The learning rate.
        disc_iters: int, default: 3
                    The number of discriminator iterations per generator training iteration.
        wasserstein_penalty: float, default: 10
                             The Wasserstein gradient penalty applied to the discriminator. See the Wasserstein GAN
                             paper for details.
        l2_penalty_generator: float, default: 1e-7
                                L2 penalty on the generator weights.
        l2_penalty_discriminator: float, default: 5e-5
                                    L2 penalty on the discriminator weights.
        temp_start: float, default: 5.0
                    The initial temperature for the Gumbel softmax.
        min_temperature: float, default: 0.5
                         The minimal temperature for the Gumbel softmax.
        temperature_decay: float, default: 1-5e-5
                           After each evaluation, the current temperature is updated as
                           current_temp := max(temperature_decay*current_temp, min_temperature)
        seed: int, default: 15
              Random seed.
        gpu_id: int or None, default: 0
                The ID of the GPU to be used for training. If None, CPU only.
        use_gumbel: bool, default: True
                Use the Gumbel softmax trick.
        
        legacy_generator: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        
        """

        self.params = {
            'noise_dim': noise_dim,
            'noise_type': noise_type,
            'Generator_Layers': generator_layers,
            'Discriminator_Layers': discriminator_layers,
            'W_Down_Generator_size': W_down_generator_size,
            'W_Down_Discriminator_size': W_down_discriminator_size,
            'l2_penalty_generator': l2_penalty_generator,
            'l2_penalty_discriminator': l2_penalty_discriminator,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'Wasserstein_penalty': wasserstein_penalty,
            'temp_start': temp_start,
            'min_temperature': min_temperature,
            'temperature_decay': temperature_decay,
            'disc_iters': disc_iters,
            'use_gumbel': use_gumbel,
            'legacy_generator': legacy_generator
        }

        assert rw_len > 1, "Random walk length must be > 1."

        # tf.reset_default_graph()
        tf.set_random_seed(seed)

        self.dir_name = dir_name
        self.N = N
        self.rw_len = rw_len

        self.walk_generator_1 = walk_generator_1
        self.walk_generator_2 = walk_generator_2
        self.walk_generator_3 = walk_generator_3

        self.privacy_walk_generator_1 = privacy_walk_generator_1
        self.privacy_walk_generator_2 = privacy_walk_generator_2
        self.privacy_walk_generator_3 = privacy_walk_generator_3

        self.noise_dim = self.params['noise_dim']
        self.G_layers = self.params['Generator_Layers']
        self.D_layers = self.params['Discriminator_Layers']
        self.tau = tf.placeholder(1.0 , shape=(), name="temperature")
        # self.g_input = tf.placeholder(tf.float32, shape=(None, self.rw_len), name="g_input")
        # self.real_data = tf.placeholder(tf.int32, shape=(None, self.rw_len), name="real_data")
        self.reward = tf.placeholder(tf.float32, shape=(None), name="reward")
        self.z = tf.placeholder(tf.float32, shape=(None, self.noise_dim), name="noise_z")
        # self.jjj = tf.placeholder(tf.float32, shape=(), name="jjj")


        # W_down and W_up for generator and discriminator
        self.W_down_generator = tf.get_variable('Generator.W_Down',
                                                shape=[self.N, self.params['W_Down_Generator_size']],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())

        self.W_down_discriminator = tf.get_variable('Discriminator.W_Down',
                                                    shape=[self.N, self.params['W_Down_Discriminator_size']],
                                                    dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())

        self.W_up = tf.get_variable("Generator.W_up", shape = [self.G_layers[-1], self.N],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        self.b_W_up = tf.get_variable("Generator.W_up_bias", dtype=tf.float32, initializer=tf.zeros_initializer,
                                      shape=self.N)

        self.generator_function = self.generator_recurrent
        self.discriminator_function = self.discriminator_recurrent


        # self.fake_inputs = self.generator_function(self.params['batch_size'], reuse=False, gumbel=use_gumbel, legacy=legacy_generator)
        self.fake_inputs = self.generator_function(self.params['batch_size'], z=self.z, reuse=False, gumbel=use_gumbel, legacy=legacy_generator)
        # self.fake_inputs_discrete = self.generate_discrete(self.params['batch_size'], reuse=True,gumbel=use_gumbel, legacy=legacy_generator)

        # Pre-fetch real random walks
        dataset = tf.data.Dataset.from_generator(self.walk_generator_1, tf.int32, [self.params['batch_size'], self.rw_len])
        #dataset_batch = dataset.prefetch(2).batch(self.params['batch_size'])
        dataset_batch = dataset.prefetch(100)
        batch_iterator = dataset_batch.make_one_shot_iterator()
        real_data = batch_iterator.get_next()

        # random sample node as random walk as real data
        real_data = tf.random_uniform([self.params['batch_size'], self.rw_len], minval=0, maxval=self.N, dtype=tf.int32)
        
        self.real_inputs_discrete = real_data

        # self.real_inputs_discrete = self.real_data
        self.real_inputs = tf.one_hot(self.real_inputs_discrete, self.N)

        self.disc_real = self.discriminator_function(self.real_inputs)
        self.disc_fake = self.discriminator_function(self.fake_inputs, reuse=True)
        
        # ====================================
        #               REWARD       
        # ====================================
        self.disc_fake_mul_reward = tf.multiply(self.disc_fake, self.reward)

        # print(self.disc_fake, tf.shape(self.disc_fake))
        # print(self.reward, tf.shape(self.reward))
        # print(self.disc_fake_mul_reward, tf.shape(self.disc_fake_mul_reward))
        # input()

        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake_mul_reward)

        # ====================================
        #           WITHOUT REWARD       
        # ====================================
        # self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        # self.gen_cost = -tf.reduce_mean(self.disc_fake)

        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(
            shape=[self.params['batch_size'], 1, 1],
            minval=0.,
            maxval=1.
        )

        self.differences = self.fake_inputs - self.real_inputs
        self.interpolates = self.real_inputs + (alpha * self.differences)
        self.gradients = tf.gradients(self.discriminator_function(self.interpolates, reuse=True), self.interpolates)[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_cost += self.params['Wasserstein_penalty'] * self.gradient_penalty

        # weight regularization; we omit W_down from regularization
        self.disc_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                     if 'Disc' in v.name
                                     and not 'W_down' in v.name]) * self.params['l2_penalty_discriminator']
        self.disc_cost += self.disc_l2_loss

        # weight regularization; we omit  W_down from regularization
        self.gen_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                     if 'Gen' in v.name
                                     and not 'W_down' in v.name]) * self.params['l2_penalty_generator']
        self.gen_cost += self.gen_l2_loss

        self.gen_params = [v for v in tf.trainable_variables() if 'Generator' in v.name]
        self.disc_params = [v for v in tf.trainable_variables() if 'Discriminator' in v.name]

        # test print gradient
        self.grad = tf.gradients(self.gen_cost, self.gen_params)

        # self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
        #                                            beta2=0.9).minimize(self.gen_cost, var_list=self.gen_params)
        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
                                                   beta2=0.9).minimize(self.gen_cost, var_list=self.gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
                                                    beta2=0.9).minimize(self.disc_cost, var_list=self.disc_params)

        # self.gen_train_op = tf.train.RMSPropOptimizer(learning_rate=self.params['learning_rate']
        #                                                 ).minimize(self.gen_cost, var_list=self.gen_params)
        # self.disc_train_op = tf.train.RMSPropOptimizer(learning_rate=self.params['learning_rate']
        #                                                 ).minimize(self.disc_cost, var_list=self.disc_params)


        if gpu_id is None:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.InteractiveSession(config=config)
        self.init_op = tf.global_variables_initializer()


    def generate_discrete(self, n_samples, reuse=True, z=None, gumbel=True, legacy=False):
        """
        Generate a random walk in index representation (instead of one hot). This is faster but prevents the gradients
        from flowing into the generator, so we only use it for evaluation purposes.

        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        
        Returns
        -------
                The generated random walks, shape [None, rw_len, N]


        """
        gumbel_output = self.generator_function(n_samples, reuse, z, gumbel=gumbel, legacy=legacy)

        return tf.argmax(gumbel_output, axis=-1)

    def generator_recurrent(self, n_samples, reuse=None, z=None, gumbel=True, legacy=False):
        """
        Generate random walks using LSTM.
        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        Returns
        -------
        The generated random walks, shape [None, rw_len, N]

        """

        with tf.variable_scope('Generator') as scope:
            if reuse is True:
                scope.reuse_variables()
            
            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.G_layers])

            # initial states h and c are randomly sampled for each lstm cell
            if z is None:
                initial_states_noise = make_noise([n_samples, self.noise_dim], self.params['noise_type'])
            else:
                initial_states_noise = z
            
            initial_states = []

            # Noise preprocessing
            for ix,size in enumerate(self.G_layers):
                if legacy: # old version to initialize LSTM. new version has less parameters and performs just as good.
                    h_intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.h_int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    h = tf.layers.dense(h_intermediate, size, name="Generator.h_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)

                    c_intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.c_int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    c = tf.layers.dense(c_intermediate, size, name="Generator.c_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                    
                else:
                    # input("default here")
                    intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    h = tf.layers.dense(intermediate, size, name="Generator.h_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                    c = tf.layers.dense(intermediate, size, name="Generator.c_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                initial_states.append((c, h))

            state = initial_states
            inputs = tf.zeros([n_samples, self.params['W_Down_Generator_size']])

            # LSTM tine steps (if rw_len=16, number of lstm layer = 16)
            outputs = []
            for i in range(self.rw_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # Get LSTM output
                output, state = self.stacked_lstm.call(inputs, state)
                
                # Blow up to dimension N using W_up
                output_bef = tf.matmul(output, self.W_up) + self.b_W_up

                # Perform Gumbel softmax to ensure gradients flow
                if gumbel:
                    output = gumbel_softmax(output_bef, temperature=self.tau, hard=True)
                    # output = gumbel_softmax(output_bef, temperature=self.tau, hard=False)
                else:
                    output = tf.nn.softmax(output_bef)

                # output = output_bef # for test

                # Back to dimension d
                inputs = tf.matmul(output, self.W_down_generator)

                outputs.append(output)

            outputs = tf.stack(outputs, axis=1)

        return outputs


    def discriminator_recurrent(self, inputs, reuse=None):
        """
        Discriminate real from fake random walks using LSTM.
        Parameters
        ----------
        inputs: tf.tensor, shape (None, rw_len, N)
                The inputs to process
        reuse: bool, default: None
               If True, discriminator variables will be reused.

        Returns
        -------
        final_score: tf.tensor, shape [None,], i.e. a scalar
                     A score measuring how "real" the input random walks are perceived.

        """
        # print(inputs)
        # input("here")
        with tf.variable_scope('Discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()

            input_reshape = tf.reshape(inputs, [-1, self.N])
            output = tf.matmul(input_reshape, self.W_down_discriminator)
            output = tf.reshape(output, [-1, self.rw_len, int(self.W_down_discriminator.get_shape()[-1])])

            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])

            output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell, inputs=tf.unstack(output, axis=1),
                                                              dtype='float32')

            last_output = output_disc[-1]

            final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")
            return final_score

    def train(self, A_orig, val_ones, val_zeros,  max_iters=50000, stopping=None, eval_transitions=15e6,
              transitions_per_iter=150000, max_patience=5, eval_every=500, plot_every=-1, save_directory="snapshots",
              model_name=None, continue_training=False, K=2, SMALLER_K=2, evaluate=False, label=None):
        """
        Parameters
        ----------
        A_orig: sparse matrix, shape: (N,N)
                Adjacency matrix of the original graph to be trained on.
        val_ones: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation edges
        val_zeros: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation non-edges
        max_iters: int, default: 50,000
                   The maximum number of training iterations if early stopping does not apply.
        stopping: float in (0,1] or None, default: None
                  The early stopping strategy. None means VAL criterion will be used (i.e. evaluation on the
                  validation set and stopping after there has not been an improvement for *max_patience* steps.
                  Set to a value in the interval (0,1] to stop when the edge overlap exceeds this threshold.
        eval_transitions: int, default: 15e6
                          The number of transitions that will be used for evaluating the validation performance, e.g.
                          if the random walk length is 5, each random walk contains 4 transitions.
        transitions_per_iter: int, default: 150000
                              The number of transitions that will be generated in one batch. Higher means faster
                              generation, but more RAM usage.
        max_patience: int, default: 5
                      Maximum evaluation steps without improvement of the validation accuracy to tolerate. Only
                      applies to the VAL criterion.
        eval_every: int, default: 500
                    Evaluate the model every X iterations.
        plot_every: int, default: -1
                    Plot the generator/discriminator losses every X iterations. Set to None or a negative number
                           to disable plotting.
        save_directory: str, default: "../snapshots"
                        The directory to save model snapshots to.
        model_name: str, default: None
                    Name of the model (will be used for saving the snapshots).
        continue_training: bool, default: False
                           Whether to start training without initializing the weights first. If False, weights will be
                           initialized.

        Returns
        -------
        log_dict: dict
                  A dictionary with the following values observed during training:
                  * The generator and discriminator losses
                  * The validation performances (ROC and AP)
                  * The edge overlap values between the generated and original graph
                  * The sampled graphs for all evaluation steps.

        """
        while True: # this while is for indentation
            if stopping == None:  # use VAL criterion
                best_performance = 0.0
                patience = max_patience
                print("**** Using VAL criterion for early stopping ****")

            else:  # use EO criterion
                assert "float" in str(type(stopping)) and stopping > 0 and stopping <= 1
                print("**** Using EO criterion of {} for early stopping".format(stopping))

            if not os.path.isdir(save_directory):
                os.makedirs(save_directory)

            if model_name is None:
                # Find the file corresponding to the lowest vacant model number to store the snapshots into.
                model_number = 0
                while os.path.exists("{}/model_best_{}.ckpt".format(save_directory, model_number)):
                    model_number += 1
                save_file = "{}/model_best_{}.ckpt".format(save_directory, model_number)
                open(save_file, 'a').close()  # touch file
            else:
                save_file = "{}/{}_best.ckpt".format(save_directory, model_name)
            print("**** Saving snapshots into {} ****".format(save_file))


            saver = tf.train.Saver()

            if not continue_training and not evaluate:
                print("**** Initializing... ****")
                self.session.run(self.init_op)
                print("**** Done.           ****")
            elif continue_training:
                print("**** Continuing training without initializing weights. ****")
                saver.restore(self.session, "snapshots/jc_reward_no_mean_rwlen60_bs16_best.ckpt")

            elif evaluate:
                print("**** Evaluate model without initiatizing weights. ****")            
                saver.restore(self.session, "snapshots/jc_reward_no_mean_rwlen60_bs16_best.ckpt")

            break

        # Validation labels
        actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros))) # 3192
        
        # Some lists to store data into.
        gen_losses = []
        disc_losses = []
        graphs = []
        val_performances = []
        eo = []
        temperature = self.params['temp_start']

        starting_time = time.time()

        transitions_per_walk = self.rw_len - 1 # 15
        # Sample lots of random walks, used for evaluation of model.
        sample_many_count = int(np.round(transitions_per_iter/transitions_per_walk)) # 10000
        # sample_many_count = self.params['batch_size']
        sample_many = self.generate_discrete(sample_many_count, reuse=True, legacy=self.params['legacy_generator']) # (10000, 16)
        n_eval_walks = eval_transitions/transitions_per_walk # 1000000
        n_eval_iters = int(np.round(n_eval_walks/sample_many_count)) # 100

        print("**** Starting training. ****")

        # structure for plot
        downsize_200_ratio_cd_performances = []
        downsize_150_ratio_cd_performances = []
        downsize_120_ratio_cd_performances = []
        downsize_20_ratio_cd_performances = []
        downsize_10_ratio_cd_performances = []
        downsize_5_ratio_cd_performances = []
        downsize_1_ratio_cd_performances = []

        downsize_200_ratio_num_of_extra_edge_performances = []
        downsize_150_ratio_num_of_extra_edge_performances = []
        downsize_120_ratio_num_of_extra_edge_performances = []
        downsize_20_ratio_num_of_extra_edge_performances = []
        downsize_10_ratio_num_of_extra_edge_performances = []
        downsize_5_ratio_num_of_extra_edge_performances = []
        downsize_1_ratio_num_of_extra_edge_performances = []

        gradient_list = []
        gen_loss_list = []
        disc_loss_list = []

        max_ari_downsize_200_ratio_graph = np.zeros((self.N, self.N), int)
        max_ari_downsize_150_ratio_graph = np.zeros((self.N, self.N), int)
        max_ari_downsize_120_ratio_graph = np.zeros((self.N, self.N), int)
        max_ari_downsize_20_ratio_graph = np.zeros((self.N, self.N), int)
        max_ari_downsize_10_ratio_graph = np.zeros((self.N, self.N), int)
        max_ari_downsize_5_ratio_graph = np.zeros((self.N, self.N), int)
        max_ari_downsize_1_ratio_graph = np.zeros((self.N, self.N), int)

        max_ari_downsize_200_val = 0
        max_ari_downsize_150_val = 0
        max_ari_downsize_120_val = 0
        max_ari_downsize_20_val = 0
        max_ari_downsize_10_val = 0
        max_ari_downsize_5_val = 0
        max_ari_downsize_1_val = 0


        G_ori = nx.from_numpy_matrix(A_orig.toarray()) # original graph for compute edge score

        for _it in range(max_iters):

            if not evaluate:

                noise_z = np_make_noise([self.params['batch_size'], self.noise_dim], self.params['noise_type'])

                # ===============================
                #       compute reward
                # ===============================
                temp_list = [] # append RW for computing edge's scorep
                fake_inputs, disc_fake = self.session.run([self.fake_inputs, self.disc_fake], feed_dict={self.tau: temperature, self.z: noise_z})
                index_fake_inputs = np.argmax(fake_inputs, axis=-1)
                
                for i in range(self.params['batch_size']):
                    temp_list.append(index_fake_inputs[i, 0:-1])
                    temp_list.append(index_fake_inputs[i, 1:])

                temp_arr = np.array(temp_list)
                temp_arr = temp_arr.reshape(int(temp_arr.shape[0]/2), 2, self.rw_len-1)

                reward = []
                for i in range(temp_arr.shape[0]):
                    per_rw_score = 1

                    # ===========================================
                    #        only positive loss mul reward
                    # ===========================================
                    if disc_fake[i] > 0:
                        for j in range(self.rw_len-1):
                            # Normal Jaccard
                            per_edge_score = [z for x, y, z in nx.jaccard_coefficient(G_ori, [(temp_arr[i, :, j][0], temp_arr[i, :, j][1])])]
                            per_rw_score += per_edge_score[0]

                            # # Density Jaccard
                            # per_edge_score = Density_jaccard(G_ori, temp_arr[i, :, j][0], temp_arr[i, :, j][1])
                            # per_rw_score += per_edge_score

                            # # No reward
                            # pass
                            
                        # reward.append(per_rw_score/(self.rw_len-1))
                        reward.append(per_rw_score)
                    else:
                        reward.append(1)

                    # ===========================================
                    #       both pos, neg loss mul reward
                    # ===========================================
                    # for j in range(self.rw_len-1):
                    #     per_edge_score = [z for x, y, z in nx.jaccard_coefficient(G_ori, [(temp_arr[i, :, j][0], temp_arr[i, :, j][1])])]
                    #     per_rw_score += per_edge_score[0]
                        
                    # reward.append(per_rw_score/(self.rw_len-1))
                    # reward.append(per_rw_score)


                # ===============================
                #       separate training
                # ===============================
                reward = np.array(reward).reshape((self.params['batch_size'], 1))
                gen_loss, grad, params, _ = self.session.run([self.gen_cost, self.grad, self.gen_params, self.gen_train_op], 
                                                    feed_dict={self.tau: temperature, self.z: noise_z, self.reward: reward})
                # gen_loss, grad, params, _ = self.session.run([self.gen_cost, self.grad, self.gen_params, self.gen_train_op], 
                #                                     feed_dict={self.tau: temperature, self.z: noise_z})

                _disc_l = []
                for _ in range(self.params['disc_iters']):
                    noise_z = np_make_noise([self.params['batch_size'], self.noise_dim], self.params['noise_type'])
                    disc_loss, _ = self.session.run(
                        [self.disc_cost, self.disc_train_op],
                        feed_dict={self.tau: temperature, self.z: noise_z}
                    )

                    _disc_l.append(disc_loss)

                gen_loss_list.append(gen_loss)
                disc_loss_list.append(np.mean(_disc_l))


                # ===============================
                #       compute gradient
                # ===============================
                gradient_count = 0
                for i in range(len(grad)):
                    gradient_count+=np.sum(np.power(grad[i], 2))
                
                gradient_list.append(gradient_count)


                print("iter: ", _it, "\tG loss: ", gen_loss, "\tD loss: ", np.mean(_disc_l), "\tGradient: ", gradient_count)

            # =============================
            # Evaluate the model's progress
            # =============================
            # if _it>0 and _it%eval_every==0:
            if _it > 1500 and _it % 50 == 0:

                # Sample lots of random walks.
                smpls = []
                for _ in range(n_eval_iters):
                    smpls.append(self.session.run(sample_many, {self.tau: 0.5}))
                # smpls.append(self.session.run(sample_many, {self.tau: 0.5}))

                # Compute score matrix
                gr = utils.score_matrix_from_random_walks(np.array(smpls).reshape([-1, self.rw_len]), self.N)
                gr = gr.tocsr()
                print(reward) # check out reward value
                print("=++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=")


                # ========================================================
                #   use original graph_from_scores function, 10 ratio
                # ========================================================
                _graph = utils.graph_from_scores(gr, int(A_orig.sum()/10))
                print("graph size: ", _graph.sum())
                extra = A_orig - _graph # seek extra edges
                extra_edge_indices = np.where(extra==-1) # seek extra edges indices
                num_of_extra_edge = extra_edge_indices[0].shape[0] # number of extra edges
                print("graph number of extra edges: ", num_of_extra_edge)
                cd_val, _ = caculate_communityDetection(_graph, label)
                print("graph CD performance: ", cd_val)
                print("================================================================")


                # ========================================================
                #   Assemble a graph from the score matrix (200% ratio)
                # ========================================================
                _graph = utils.young_graph_from_scores(gr, int(A_orig.sum()*(200/100)))
                print("200 FD graph size: ", _graph.sum())
                extra = A_orig - _graph # seek extra edges
                extra_edge_indices = np.where(extra==-1) # seek extra edges indices
                downsize_200_ratio_num_of_extra_edge = extra_edge_indices[0].shape[0] # number of extra edges
                print("200 FD graph number of extra edges: ", downsize_200_ratio_num_of_extra_edge)
                downsize_200_ratio_cd_val, _ = caculate_communityDetection(_graph, label)
                print("200 FD graph CD performance: ", downsize_200_ratio_cd_val)
                if downsize_200_ratio_cd_val > max_ari_downsize_200_val:
                    max_ari_downsize_200_val = downsize_200_ratio_cd_val
                    max_ari_downsize_200_ratio_graph = _graph.copy()
                print("================================================================")


                # ========================================================
                #   Assemble a graph from the score matrix (150% ratio)
                # ========================================================
                _graph = utils.young_graph_from_scores(gr, int(A_orig.sum()*(150/100)))
                print("150 FD graph size: ", _graph.sum())
                extra = A_orig - _graph # seek extra edges
                extra_edge_indices = np.where(extra==-1) # seek extra edges indices
                downsize_150_ratio_num_of_extra_edge = extra_edge_indices[0].shape[0] # number of extra edges
                print("150 FD graph number of extra edges: ", downsize_150_ratio_num_of_extra_edge)
                downsize_150_ratio_cd_val, _ = caculate_communityDetection(_graph, label)
                print("150 FD graph CD performance: ", downsize_150_ratio_cd_val)
                if downsize_150_ratio_cd_val > max_ari_downsize_150_val:
                    max_ari_downsize_150_val = downsize_150_ratio_cd_val
                    max_ari_downsize_150_ratio_graph = _graph.copy()
                print("================================================================")


                # ========================================================
                #   Assemble a graph from the score matrix (120% ratio)
                # ========================================================
                _graph = utils.young_graph_from_scores(gr, int(A_orig.sum()*(120/100)))
                print("120 FD graph size: ", _graph.sum())
                extra = A_orig - _graph # seek extra edges
                extra_edge_indices = np.where(extra==-1) # seek extra edges indices
                downsize_120_ratio_num_of_extra_edge = extra_edge_indices[0].shape[0] # number of extra edges
                print("120 FD graph number of extra edges: ", downsize_120_ratio_num_of_extra_edge)
                downsize_120_ratio_cd_val, _ = caculate_communityDetection(_graph, label)
                print("120 FD graph CD performance: ", downsize_120_ratio_cd_val)
                if downsize_120_ratio_cd_val > max_ari_downsize_120_val:
                    max_ari_downsize_120_val = downsize_120_ratio_cd_val
                    max_ari_downsize_120_ratio_graph = _graph.copy()
                print("================================================================")


                # ========================================================
                #   Assemble a graph from the score matrix (20% ratio)
                # ========================================================
                _graph = utils.young_graph_from_scores(gr, int(A_orig.sum()*(20/100)))
                print("20 FD graph size: ", _graph.sum())
                extra = A_orig - _graph # seek extra edges
                extra_edge_indices = np.where(extra==-1) # seek extra edges indices
                downsize_20_ratio_num_of_extra_edge = extra_edge_indices[0].shape[0] # number of extra edges
                print("20 FD graph number of extra edges: ", downsize_20_ratio_num_of_extra_edge)
                downsize_20_ratio_cd_val, _ = caculate_communityDetection(_graph, label)
                print("20 FD graph CD performance: ", downsize_20_ratio_cd_val)
                if downsize_20_ratio_cd_val > max_ari_downsize_20_val:
                    max_ari_downsize_20_val = downsize_20_ratio_cd_val
                    max_ari_downsize_20_ratio_graph = _graph.copy()
                print("================================================================")


                # ========================================================
                #   Assemble a graph from the score matrix (10% ratio)
                # ========================================================
                _graph = utils.young_graph_from_scores(gr, int(A_orig.sum()*(10/100)))
                print("10 FD graph size: ", _graph.sum())
                extra = A_orig - _graph # seek extra edges
                extra_edge_indices = np.where(extra==-1) # seek extra edges indices
                downsize_10_ratio_num_of_extra_edge = extra_edge_indices[0].shape[0] # number of extra edges
                print("10 FD graph number of extra edges: ", downsize_10_ratio_num_of_extra_edge)
                downsize_10_ratio_cd_val, _ = caculate_communityDetection(_graph, label)
                print("10 FD graph CD performance: ", downsize_10_ratio_cd_val)
                if downsize_10_ratio_cd_val > max_ari_downsize_10_val:
                    max_ari_downsize_10_val = downsize_10_ratio_cd_val
                    max_ari_downsize_10_ratio_graph = _graph.copy()
                print("================================================================")


                # ========================================================
                #   Assemble a graph from the score matrix (5% ratio)
                # ========================================================
                _graph = utils.young_graph_from_scores(gr, int(A_orig.sum()*(5/100)))
                print("5 FD graph size: ", _graph.sum())
                extra = A_orig - _graph # seek extra edges
                extra_edge_indices = np.where(extra==-1) # seek extra edges indices
                downsize_5_ratio_num_of_extra_edge = extra_edge_indices[0].shape[0] # number of extra edges
                print("5 FD graph number of extra edges: ", downsize_5_ratio_num_of_extra_edge)
                downsize_5_ratio_cd_val, _ = caculate_communityDetection(_graph, label)
                print("5 FD graph CD performance: ", downsize_5_ratio_cd_val)
                if downsize_5_ratio_cd_val > max_ari_downsize_5_val:
                    max_ari_downsize_5_val=  downsize_5_ratio_cd_val
                    max_ari_downsize_5_ratio_graph = _graph.copy()
                print("================================================================")


                # ========================================================
                #   Assemble a graph from the score matrix (1% ratio)
                # ========================================================
                _graph = utils.young_graph_from_scores(gr, int(A_orig.sum()*(1/100)))
                print("1 FD graph size: ", _graph.sum())
                extra = A_orig - _graph # seek extra edges
                extra_edge_indices = np.where(extra==-1) # seek extra edges indices
                downsize_1_ratio_num_of_extra_edge = extra_edge_indices[0].shape[0] # number of extra edges
                print("1 FD graph number of extra edges: ", downsize_1_ratio_num_of_extra_edge)
                downsize_1_ratio_cd_val, _ = caculate_communityDetection(_graph, label)
                print("1 FD graph CD performance: ", downsize_1_ratio_cd_val)
                if downsize_1_ratio_cd_val > max_ari_downsize_1_val:
                    max_ari_downsize_1_val = downsize_1_ratio_cd_val
                    max_ari_downsize_1_ratio_graph = _graph.copy()
                print("=++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=")

                # record community detection performance
                downsize_200_ratio_cd_performances.append(downsize_200_ratio_cd_val)
                downsize_150_ratio_cd_performances.append(downsize_150_ratio_cd_val)
                downsize_120_ratio_cd_performances.append(downsize_120_ratio_cd_val)
                downsize_20_ratio_cd_performances.append(downsize_20_ratio_cd_val)
                downsize_10_ratio_cd_performances.append(downsize_10_ratio_cd_val)
                downsize_5_ratio_cd_performances.append(downsize_5_ratio_cd_val)
                downsize_1_ratio_cd_performances.append(downsize_1_ratio_cd_val)

                # record each ratio, number of extra edges
                downsize_200_ratio_num_of_extra_edge_performances.append(downsize_200_ratio_num_of_extra_edge)
                downsize_150_ratio_num_of_extra_edge_performances.append(downsize_150_ratio_num_of_extra_edge)
                downsize_120_ratio_num_of_extra_edge_performances.append(downsize_120_ratio_num_of_extra_edge)
                downsize_20_ratio_num_of_extra_edge_performances.append(downsize_20_ratio_num_of_extra_edge)
                downsize_10_ratio_num_of_extra_edge_performances.append(downsize_10_ratio_num_of_extra_edge)
                downsize_5_ratio_num_of_extra_edge_performances.append(downsize_5_ratio_num_of_extra_edge)
                downsize_1_ratio_num_of_extra_edge_performances.append(downsize_1_ratio_num_of_extra_edge)


                os.makedirs("pic/"+self.dir_name, exist_ok=True)
                np.save("pic/"+self.dir_name+"/gradients", np.array(gradient_list))
                np.save("pic/"+self.dir_name+"/gen_loss", np.array(gen_loss_list))
                np.save("pic/"+self.dir_name+"/disc_loss_list", np.array(disc_loss_list))


                if _it > 0 and _it % 5000 == 0:
                    os.makedirs("pic/"+self.dir_name, exist_ok=True)
                    np.save("pic/"+self.dir_name+"/downsize_200_ratio_cd_performances", np.array(downsize_200_ratio_cd_performances))
                    np.save("pic/"+self.dir_name+"/downsize_150_ratio_cd_performances", np.array(downsize_150_ratio_cd_performances))
                    np.save("pic/"+self.dir_name+"/downsize_120_ratio_cd_performances", np.array(downsize_120_ratio_cd_performances))
                    np.save("pic/"+self.dir_name+"/downsize_20_ratio_cd_performances", np.array(downsize_20_ratio_cd_performances))
                    np.save("pic/"+self.dir_name+"/downsize_10_ratio_cd_performances", np.array(downsize_10_ratio_cd_performances))
                    np.save("pic/"+self.dir_name+"/downsize_5_ratio_cd_performances", np.array(downsize_5_ratio_cd_performances))
                    np.save("pic/"+self.dir_name+"/downsize_1_ratio_cd_performances", np.array(downsize_1_ratio_cd_performances))

                    np.save("pic/"+self.dir_name+"/downsize_200_ratio_num_of_extra_edge_performances", np.array(downsize_200_ratio_num_of_extra_edge_performances))
                    np.save("pic/"+self.dir_name+"/downsize_150_ratio_num_of_extra_edge_performances", np.array(downsize_150_ratio_num_of_extra_edge_performances))
                    np.save("pic/"+self.dir_name+"/downsize_120_ratio_num_of_extra_edge_performances", np.array(downsize_120_ratio_num_of_extra_edge_performances))
                    np.save("pic/"+self.dir_name+"/downsize_20_ratio_num_of_extra_edge_performances", np.array(downsize_20_ratio_num_of_extra_edge_performances))
                    np.save("pic/"+self.dir_name+"/downsize_10_ratio_num_of_extra_edge_performances", np.array(downsize_10_ratio_num_of_extra_edge_performances))
                    np.save("pic/"+self.dir_name+"/downsize_5_ratio_num_of_extra_edge_performances", np.array(downsize_5_ratio_num_of_extra_edge_performances))
                    np.save("pic/"+self.dir_name+"/downsize_1_ratio_num_of_extra_edge_performances", np.array(downsize_1_ratio_num_of_extra_edge_performances))

            if _it > 0 and _it % 2000 == 0:
                # Update Gumbel temperature
                temperature = np.maximum(self.params['temp_start'] * np.exp(-(1-self.params['temperature_decay']) * _it),
                                         self.params['min_temperature'])                                

                
        os.makedirs('pic/' + self.dir_name, exist_ok=True)
        #========================================
        #   model CD performance plot
        #========================================
        x = np.arange(1550, max_iters, 50)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(x, downsize_200_ratio_cd_performances, color='blue', marker='D', label='200%')
        plt.plot(x, downsize_150_ratio_cd_performances, color='blue', marker='h', label='150%')
        plt.plot(x, downsize_120_ratio_cd_performances, color='blue', marker='x', label='120%')
        plt.plot(x, downsize_20_ratio_cd_performances, color='red', marker='o', label='20%')
        plt.plot(x, downsize_10_ratio_cd_performances, color='red', marker='*', label='10%')
        plt.plot(x, downsize_5_ratio_cd_performances, color='red', marker='v', label='5%')
        plt.plot(x, downsize_1_ratio_cd_performances, color='red', marker='+', label='1%')
        plt.legend(loc='upper right')
        plt.xlabel("Epochs")
        plt.ylabel("Result")
        plt.title("Community detection Performance")
        plt.savefig("pic/"+self.dir_name+"/cd_result.png", dpi=600, format="png")

        #========================================
        #       number of extra edge plot
        #========================================
        x = np.arange(1550, max_iters, 50)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(x, downsize_200_ratio_num_of_extra_edge_performances, color='blue', marker='D', label='200%')
        plt.plot(x, downsize_150_ratio_num_of_extra_edge_performances, color='blue', marker='h', label='150%')
        plt.plot(x, downsize_120_ratio_num_of_extra_edge_performances, color='blue', marker='x', label='120%')
        plt.plot(x, downsize_20_ratio_num_of_extra_edge_performances, color='red', marker='o', label='20%')
        plt.plot(x, downsize_10_ratio_num_of_extra_edge_performances, color='red', marker='*', label='10%')
        plt.plot(x, downsize_5_ratio_num_of_extra_edge_performances, color='red', marker='v', label='5%')
        plt.plot(x, downsize_1_ratio_num_of_extra_edge_performances, color='red', marker='+', label='1%')
        plt.legend(loc='upper right')
        plt.xlabel("Epochs")
        plt.ylabel("Result")
        plt.title("Number of extra edge")
        plt.savefig("pic/"+self.dir_name+"/number_of_extra_edge.png", dpi=600, format="png")

        #========================================
        #   gradient plot
        #========================================
        x = np.arange(0, max_iters)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(x, gradient_list, color='blue', marker='o', label='gradients')
        plt.plot(x, gen_loss_list, color='blue', marker='s', label='gen_loss')
        plt.plot(x, disc_loss_list, color='blue', marker='x', label='disc_loss')
        plt.legend(loc='lower right')
        plt.xlabel("Epochs")
        plt.ylabel("Result")
        plt.savefig("pic/"+self.dir_name+"/gradients_result.png", dpi=600, format="png")

        #========================================
        #   save metric list as npy
        #========================================
        np.save("pic/"+self.dir_name+"/downsize_200_ratio_cd_performances", np.array(downsize_200_ratio_cd_performances))
        np.save("pic/"+self.dir_name+"/downsize_150_ratio_cd_performances", np.array(downsize_150_ratio_cd_performances))
        np.save("pic/"+self.dir_name+"/downsize_120_ratio_cd_performances", np.array(downsize_120_ratio_cd_performances))
        np.save("pic/"+self.dir_name+"/downsize_20_ratio_cd_performances", np.array(downsize_20_ratio_cd_performances))
        np.save("pic/"+self.dir_name+"/downsize_10_ratio_cd_performances", np.array(downsize_10_ratio_cd_performances))
        np.save("pic/"+self.dir_name+"/downsize_5_ratio_cd_performances", np.array(downsize_5_ratio_cd_performances))
        np.save("pic/"+self.dir_name+"/downsize_1_ratio_cd_performances", np.array(downsize_1_ratio_cd_performances))

        np.save("pic/"+self.dir_name+"/downsize_200_ratio_num_of_extra_edge_performances", np.array(downsize_200_ratio_num_of_extra_edge_performances))
        np.save("pic/"+self.dir_name+"/downsize_150_ratio_num_of_extra_edge_performances", np.array(downsize_150_ratio_num_of_extra_edge_performances))
        np.save("pic/"+self.dir_name+"/downsize_120_ratio_num_of_extra_edge_performances", np.array(downsize_120_ratio_num_of_extra_edge_performances))
        np.save("pic/"+self.dir_name+"/downsize_20_ratio_num_of_extra_edge_performances", np.array(downsize_20_ratio_num_of_extra_edge_performances))
        np.save("pic/"+self.dir_name+"/downsize_10_ratio_num_of_extra_edge_performances", np.array(downsize_10_ratio_num_of_extra_edge_performances))
        np.save("pic/"+self.dir_name+"/downsize_5_ratio_num_of_extra_edge_performances", np.array(downsize_5_ratio_num_of_extra_edge_performances))
        np.save("pic/"+self.dir_name+"/downsize_1_ratio_num_of_extra_edge_performances", np.array(downsize_1_ratio_num_of_extra_edge_performances))

        # np.save("pic/"+self.dir_name+"/max_ari_downsize_200_ratio_graph", np.array(max_ari_downsize_200_ratio_graph))
        # np.save("pic/"+self.dir_name+"/max_ari_downsize_150_ratio_graph", np.array(max_ari_downsize_150_ratio_graph))
        # np.save("pic/"+self.dir_name+"/max_ari_downsize_120_ratio_graph", np.array(max_ari_downsize_120_ratio_graph))
        # np.save("pic/"+self.dir_name+"/max_ari_downsize_20_ratio_graph", np.array(max_ari_downsize_20_ratio_graph))
        # np.save("pic/"+self.dir_name+"/max_ari_downsize_10_ratio_graph", np.array(max_ari_downsize_10_ratio_graph))
        # np.save("pic/"+self.dir_name+"/max_ari_downsize_5_ratio_graph", np.array(max_ari_downsize_5_ratio_graph))
        # np.save("pic/"+self.dir_name+"/max_ari_downsize_1_ratio_graph", np.array(max_ari_downsize_1_ratio_graph))

        scipy.sparse.save_npz("pic/"+self.dir_name+"/max_ari_downsize_200_ratio_graph.npz", sparse.csr_matrix(np.array(max_ari_downsize_200_ratio_graph)))
        scipy.sparse.save_npz("pic/"+self.dir_name+"/max_ari_downsize_150_ratio_graph.npz", sparse.csr_matrix(np.array(max_ari_downsize_150_ratio_graph)))
        scipy.sparse.save_npz("pic/"+self.dir_name+"/max_ari_downsize_120_ratio_graph.npz", sparse.csr_matrix(np.array(max_ari_downsize_120_ratio_graph)))
        scipy.sparse.save_npz("pic/"+self.dir_name+"/max_ari_downsize_20_ratio_graph.npz", sparse.csr_matrix(np.array(max_ari_downsize_20_ratio_graph)))
        scipy.sparse.save_npz("pic/"+self.dir_name+"/max_ari_downsize_10_ratio_graph.npz", sparse.csr_matrix(np.array(max_ari_downsize_10_ratio_graph)))
        scipy.sparse.save_npz("pic/"+self.dir_name+"/max_ari_downsize_5_ratio_graph.npz", sparse.csr_matrix(np.array(max_ari_downsize_5_ratio_graph)))
        scipy.sparse.save_npz("pic/"+self.dir_name+"/max_ari_downsize_1_ratio_graph.npz", sparse.csr_matrix(np.array(max_ari_downsize_1_ratio_graph)))




        #------------------------------------------------------------------------
        if stopping is None:
            saver.save(self.session, save_file)

        log_dict = {"disc_losses": disc_losses, 'gen_losses': gen_losses}

        return 1


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.

    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".

    Returns
    -------
    noise tensor

    """

    if type == "Gaussian":
        noise = tf.random_normal(shape)
    elif type == 'Uniform':
        noise = tf.random_uniform(shape, minval=-1, maxval=1)
    else:
        print("ERROR: Noise type {} not supported".format(type))
    return noise

def np_make_noise(shape, type="Gaussian"):
    '''
    Generate random noise in numpy object
    '''
    if type == "Gaussian":
        noise = np.random.normal(size=shape)
    elif type == 'Uniform':
        noise = np.random.uniform(size=shape, low=-1, high=1)
    else:
        print("ERROR: Noise type {} not supported".format(type))
    return noise

def sample_gumbel(shape, eps=1e-20):
    """
    Sample from a uniform Gumbel distribution. Code by Eric Jang available at
    http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    Parameters
    ----------
    shape: Shape of the Gumbel noise
    eps: Epsilon for numerical stability.

    Returns
    -------
    Noise drawn from a uniform Gumbel distribution.

    """
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def comput_degree(vector):
    """Return the degree of the node.
    Args:
        vector: node representation as adj vector
    """
    return len(vector[vector>0.5])

def compute_privacy(A, N, k):
    '''Compute percentage of privacy of A
    Args:
        A: graph representation as adjacency matrix
    '''
    # print("starting compute percentage of privacy of matrix")
    A = np.array(A)
    arr = A.sum(axis=1)
    # print(arr)
    uni = np.unique(arr)
    list_a = []
    scores = 0
    for i in uni:
        count = 0
        for j in arr:
            if i==j:
                count+=1
        list_a.append(count)
    arr_a = np.array(list_a)
    scores = sum([x for x in arr_a if x >= k])/N
    return scores

def caculate_linkPrediction(T1_matrix, val_ones, val_zeros):
    #----------------------
    # 同時參考上下三角的資訊
    #----------------------
    T1_matrix = (T1_matrix + T1_matrix.T) / 2
    thrs = 0.0000001
    numOfNodes = T1_matrix.shape[0]
    # draw graph
    T1_G = nx.Graph()
    for i in range(numOfNodes):
        T1_G.add_node(i)

    for i in range(T1_matrix.shape[0]):
        for j in range(i+1, T1_matrix.shape[1]):
            if T1_matrix[i, j] > thrs:
                T1_G.add_edge(i, j, weight=T1_matrix[i, j])

    #------------------------------------
    # measure positive and negative edge
    #------------------------------------
    pos_node1 = []
    pos_node2 = []
    neg_node1 = []
    neg_node2 = []

    for i in range(T1_matrix.shape[0]):
        for j in range(i+1, T1_matrix.shape[1]):
            if T1_matrix[i, j] > thrs:
                pos_node1.append(i)
                pos_node2.append(j)
            else: 
                neg_node1.append(i)
                neg_node2.append(j)

    pos_dict = {"node1": pos_node1, "node2": pos_node2}
    neg_dict = {"node1": neg_node1, "node2": neg_node2}
    pos_df = pd.DataFrame(pos_dict)
    neg_df = pd.DataFrame(neg_dict)
    # print("total pos edge len: ", pos_df.shape[0])
    # print("total neg edge len: ", neg_df.shape[0])
    # input()

    # train_pos_df, test_pos_df = train_test_split(pos_df, test_size=0.0)
    # train_neg_df, test_neg_df = train_test_split(neg_df, test_size=0.0)

    train_pos_df = pos_df
    train_neg_df = neg_df

    '''看正負樣本各自數量決定要取多少筆數, 要選小的,
        不然會indexer out of bounds
    '''
    numOfSamples = min(train_pos_df.shape[0], train_neg_df.shape[0])
    # print(train_pos_df.shape[0], "\t", train_neg_df.shape[0])
    # numOfSamples = train_pos_df.shape[0]
    # numOfSamples = 200

    lp_X_train = np.zeros((2*numOfSamples, 1), float)
    lp_y_train = np.zeros((2*numOfSamples, 1), float)

    pos_cn_list = []
    neg_cn_list = []

    '''正的筆副得多時 會抱錯, 父的會超過比數因為沒這麼多筆ˋ
    '''
    for i in range(numOfSamples):
        # positive sample common neighbors
        pos_cn = len(list(
            nx.common_neighbors(T1_G, train_pos_df.iloc[i, 0], train_pos_df.iloc[i, 1])))
        pos_cn_list.append(pos_cn)

        # negative sample common neighbors
        neg_cn = len(list(
            nx.common_neighbors(T1_G, train_neg_df.iloc[i, 0], train_neg_df.iloc[i, 1])))
        neg_cn_list.append(neg_cn)


    lp_X_train[0:numOfSamples, 0] = pos_cn_list
    lp_X_train[numOfSamples:, 0] = neg_cn_list
    # print("pos cn feature: ", pos_cn_list[0:10])
    # print("neg cn feature: ", neg_cn_list[0:10])

    lp_y_train[0:numOfSamples] = 1
    lp_y_train[numOfSamples:] = 0

    # random permutation
    indices = np.random.permutation(lp_X_train.shape[0])
    lp_X_train = lp_X_train[indices]
    lp_y_train = lp_y_train[indices]

    #-------------------------------------- 
    #       link prediction training
    #-------------------------------------- 
    xgbc = XGBClassifier()
    xgbc.fit(lp_X_train.reshape((lp_X_train.shape[0], 1)), 
                lp_y_train.reshape((lp_y_train.shape[0])))

    #-------------------------------------- 
    #              TESTING
    #-------------------------------------- 
    lp_X_test = np.zeros((len(val_ones)+len(val_zeros), 1), float)
    pos_cn_list = []
    neg_cn_list = []
    pos_y_true = []
    neg_y_true = []

    for i in range(len(val_ones)):
        pos_cn = len(list(nx.common_neighbors(T1_G, val_ones[i, 0], val_ones[i, 1])))
        pos_cn_list.append(pos_cn)

        neg_cn = len(list(nx.common_neighbors(T1_G, val_zeros[i, 0], val_zeros[i, 1])))
        neg_cn_list.append(neg_cn)

        pos_y_true.append(1.0)
        neg_y_true.append(0.0)

    lp_X_test[0:len(val_ones), 0] = pos_cn_list
    lp_X_test[len(val_ones):, 0] = neg_cn_list

    y_predict = xgbc.predict(lp_X_test.reshape((lp_X_test.shape[0], 1)))
    y_true = np.concatenate((np.array(pos_y_true), np.array(neg_y_true)), axis=0)

    # print("y_predict ==> ", y_predict)
    # print("y_true ==> ", y_true)
    # input()

    f1_score = metrics.f1_score(y_true, y_predict)

    return f1_score

def caculate_communityDetection(gen_graph, label):
    partition = community.best_partition(nx.from_numpy_matrix(gen_graph))
    num_of_partition = max(partition.values())
    keys, values = zip(*partition.items())
    pred = np.array(list(values))

    return adjusted_rand_score(label, pred), num_of_partition

def Density_jaccard(graph, node1, node2):
    cn = sorted(nx.common_neighbors(graph, node1, node2))
    node1_neighbors = [n for n in graph.neighbors(node1)]
    node2_neighbors = [n for n in graph.neighbors(node2)]
    union = list(set(node1_neighbors).union(set(node2_neighbors)))
    
    jaccard = len(cn) / len(union)
    temp_g = graph.subgraph(union)
    density = nx.density(temp_g)*10
    if density==0:
        density = nx.density(graph)*10

    return jaccard/density