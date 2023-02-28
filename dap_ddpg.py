# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from random import uniform

tf.reset_default_graph()

render = False

class DDPG(object):
    def __init__(self):
        self.lr_a = 1e-5 # learning rate for actor
        self.lr_c = 1e-4 # learning rate for critic
        self.lr_co = 1e-4 # learning rate for cost
        self.lr_lam = 1e-4 # learning rate for dual variable
        self.batch_size = 2 # batch size for training
        self.memory_size = 10000 # size for memory buffer, enough for FL training
        self.gamma = 0.99 # discount factor
        self.beta = 0.001 # soft replacement
        self.action_shape = 2 # i.e., learning rate and training epochs
        self.state_shape = 3 # i.e., loss, accuracy and f1-score
        self.cost_constraint = 1 # i.e., E(t)
        self.memory = np.zeros((self.memory_size, self.state_shape * 2 + self.action_shape + 1), dtype=np.float32) # empty memory buffer
        self.sess = tf.Session() # initialize tensorflow session
        
        # placeholder for current state, next state, reward and cost
        self.state = tf.placeholder(tf.float32, [None, self.state_shape], 's')
        self.next_state = tf.placeholder(tf.float32, [None, self.state_shape], 's_')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'r')
        self.cost = tf.placeholder(tf.float32, [None, 1], 'c')
        
        # dual variable
        self.lam = tf.Variable(0.)
        
        # build the actor network and target actor network
        with tf.variable_scope('Actor'):
            self.a = self.build_actor_network(self.state, scope='eval', trainable=True)
            a_ = self.build_actor_network(self.next_state, scope='target', trainable=False)
            
        # build the critic network and target critic network    
        with tf.variable_scope('Critic'):
            q = self.build_crtic_network(self.state, self.a, scope='eval', trainable=True)
            q_ = self.build_crtic_network(self.next_state, a_, scope='target', trainable=False)
        
        # build the cost network and target cost network, similar to critic
        with tf.variable_scope('Cost'):
            c = self.build_crtic_network(self.state, self.a, scope='eval', trainable=True)
            c_ = self.build_crtic_network(self.next_state, a_, scope='target', trainable=False)
            
        # initialize the network parameters
        self.actor_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.actor_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        
        self.critic_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.critic_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        
        self.cost_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Cost/eval')
        self.cost_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Cost/target')

        # soft-update
        self.soft_replace = [[tf.assign(at, (1-self.beta)*at + self.beta*ae), \
                              tf.assign(ct, (1-self.beta)*ct + self.beta*ce), \
                              tf.assign(ct1, (1-self.beta)*ct1 + self.beta*ce1)]
            for at, ae, ct, ce, ct1, ce1 in zip(self.actor_target_params, self.actor_eval_params, \
                                                self.critic_target_params, self.critic_eval_params, \
                                                self.cost_target_params, self.cost_eval_params)]
        
        # target Q value, TD error and critic network train operation
        q_target = self.reward + self.gamma * q_
        self.critic_td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.critic_train_op = tf.train.AdamOptimizer(self.lr_c)
        self.critic_train = self.critic_train_op.minimize(self.critic_td_error, var_list = self.critic_eval_params)
        
        # target cost value, TD error and cost critic network train operation
        c_target = self.cost + self.gamma * c_
        self.cost_td_error = tf.losses.mean_squared_error(labels=c_target, predictions=c)
        self.cost_train_op = tf.train.AdamOptimizer(self.lr_co)
        self.cost_train = self.cost_train_op.minimize(self.cost_td_error, var_list = self.cost_eval_params)
        
        # actor loss and actor network train operation
        self.actor_loss = -tf.reduce_mean(q - self.lam*c)
        self.actor_train_op = tf.train.AdamOptimizer(self.lr_a)
        self.actor_train = self.actor_train_op.minimize(self.actor_loss, var_list=self.actor_eval_params)
        
        # dual variable loss
        E_t = self.cost_constraint + uniform(-0.1, 0.1) # cost constant can also be time-varying
        self.lam_update = tf.reduce_mean(c - E_t)
   
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
        
        # define the saver
        self.saver = tf.train.Saver()


    
    def build_actor_network(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 10, activation=tf.nn.tanh, name='l1', trainable=trainable)
            l2 = tf.layers.dense(l1, 5, activation=tf.nn.tanh, name='l2', trainable=trainable)
            a = tf.layers.dense(l2, self.action_shape, activation=tf.nn.tanh, name='a', trainable=trainable)     
        return a


    def build_crtic_network(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.state_shape, 5], trainable = trainable)
            w1_a = tf.get_variable('w1_a', [self.action_shape, 5], trainable = trainable)
            b1 = tf.get_variable('b1', [1, 5], trainable = trainable)
            l1 = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1
            l2 = tf.layers.dense(l1, 5, activation=tf.nn.tanh, trainable = trainable)
            q = tf.layers.dense(l2, 1, trainable = trainable)
        return q


    def choose_action(self, s):
        a = self.sess.run(self.a, {self.state: s[np.newaxis, :]})[0]
        return a
    
    
    def learn(self):
        # sample
        record_num = len(self.memory)
        indices = np.random.choice(record_num, size=self.batch_size)
        batch_transition = self.memory[indices, :]
        
        # divide batch into [states, actions, rewards, costs, next_states]
        batch_states = batch_transition[:, :self.state_shape]
        batch_actions = batch_transition[:, self.state_shape: self.state_shape+self.action_shape]
        batch_rewards = batch_transition[:, self.state_shape+self.action_shape: self.state_shape+self.action_shape+1]
        batch_costs = batch_transition[:, self.state_shape+self.action_shape+1: self.state_shape+self.action_shape+2]
        batch_next_state = batch_transition[:, -self.state_shape:]
        
        # train critic and get loss value
        self.sess.run(self.critic_train, {self.state: batch_states, self.a: batch_actions,
                                    self.reward: batch_rewards, self.next_state: batch_next_state})       
        critic_td_error_value = self.sess.run(self.critic_td_error, {self.state: batch_states, self.a: batch_actions,
                                    self.reward: batch_rewards, self.next_state: batch_next_state})
        
        # train cost and get loss value
        self.sess.run(self.cost_train, {self.state: batch_states, self.a: batch_actions,
                                    self.cost: batch_costs, self.next_state: batch_next_state})       
        cost_td_error_value = self.sess.run(self.cost_td_error, {self.state: batch_states, self.a: batch_actions,
                                    self.cost: batch_costs, self.next_state: batch_next_state})
        
        # train actor and get loss value
        self.sess.run(self.actor_train, {self.state: batch_states})  
        actor_loss_value = self.sess.run(self.actor_loss, {self.state: batch_states, self.a: batch_actions})
        
        # update three target networks
        self.sess.run(self.soft_replace)
        
        # update dual variable
        self.lam.assign(self.lr_lam*self.lam_update)
        self.lam.assign(tf.maximum(self.lam, 0.))
        
        return actor_loss_value, critic_td_error_value, cost_td_error_value
        
    
    # save ddpg model
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    # restore ddpg model
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)