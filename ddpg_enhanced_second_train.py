import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

from sklearn.metrics import f1_score
from numpy import argmax
from random import uniform
from copy import deepcopy
from dap_ddpg import DDPG
from build_temp_model import build_cnn_model as cnn
from build_temp_model import build_logistic_model as log
from build_temp_model import build_resnet_model as res

import warnings
warnings.filterwarnings("ignore")

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



def load_server_test_data():
    # modify here to load other test data, i.e., fashion-mnist and femnist
    (_, _),(x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_true = y_test 
    y_test = keras.utils.to_categorical(y_test, 10) 
    return x_test, y_test, y_true


def load_client_data(i):
    # load data from disk
    all_data = np.load('users/user_'+str(i)+'/data.npz')
    x_train, y_train = all_data['arr_0'], all_data['arr_1']
    
    # operations on samples
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_train = x_train / 255
    y_true = y_train  # record to compute F1-score
    y_train = keras.utils.to_categorical(y_train, 10)
    x_test, y_test = x_train, y_train # train data is also the test data
    
    return x_train, y_train, x_test, y_test, y_true


def compute_F1_socre(model, test_samples, true_labels):
    pred_labels = model.predict(test_samples, batch_size=32, verbose=0)
    pred_labels = [argmax(y) for y in pred_labels]
    f1 = f1_score(true_labels, pred_labels, average='micro')
    return f1


def average_model(user_num, weights, t, model_class):
    if model_class == "cnn":
        temp_model_weight, model_len = cnn()
    elif model_class == "logistic":
        temp_model_weight, model_len = log()
    elif model_class == "resnet":
        temp_model_weight, model_len = res()
    
    for i in range(user_num):
        user_model_weight = load_model('users/user_'+str(i)+'/'+str(t+1)+'_th.h5').get_weights()
        for j in range(model_len):
            weighted_user_weight = [element * weights[i] for element in user_model_weight[j]]
            temp_model_weight[j] = temp_model_weight[j] +  weighted_user_weight
    for i in range(model_len):
        temp_model_weight[i] = temp_model_weight[i] / sum(weights[:user_num])
    
    ## backfill and save global models
    updated_global_model = load_model('server/W_0_th.h5')
    updated_global_model.set_weights(temp_model_weight)
    updated_global_model.save('server/W_'+str(t+1)+'_th.h5')
    
    # server tests global model
    server_x_test, server_y_test, server_y_true = load_server_test_data()
    loss, acc, fs = observe(updated_global_model, server_x_test, server_y_test, server_y_true)
    return loss, acc, fs


def observe(model, x_test, y_test, y_true):
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    fs = compute_F1_socre(model, x_test, y_true)
    return loss, acc, fs


def save_csv(filedata, filepath):
    temp=pd.DataFrame(filedata)
    temp.to_csv(filepath, encoding='gbk', header=0, index=None) 


# define parameters
base_learning_rate = 10 # bottom number of local learining rate
base_iteration = 20 # base number of local training epochs
user_num = 5 # the number of clients
start_iteration = 0 # the start of local training
iteration_num = 2 # the number of training rounds sustained in local training
second_batch_size = 32 # the batch size for local training
xi1, xi2, xi3 = 1, 1, 1 # coefficient of reward function

# define the recorders
empty_recoder = [[] for _ in range(user_num)]
local_action0, local_action1= deepcopy(empty_recoder), deepcopy(empty_recoder) # local action record
local_loss, local_acc, local_fs = deepcopy(empty_recoder), deepcopy(empty_recoder), deepcopy(empty_recoder) # local state record
reward, cost = deepcopy(empty_recoder), deepcopy(empty_recoder) # local reward record and cost record
learning_rate, epoch_num = deepcopy(empty_recoder), deepcopy(empty_recoder) # local hypermater record, conputed from actions
actor_loss, critic_loss, cost_loss = deepcopy(empty_recoder), deepcopy(empty_recoder), deepcopy(empty_recoder) # local ddpg train loss record
global_loss, global_acc, global_fs = [], [], [] # recorded by server

# define the aggregation weight, i.e., the size of lacal data
weights = [592, 674, 595, 613, 584, 542, 591, 626, 585, 594]

# substitue ddpg model
ddpg = DDPG()

for t in range(start_iteration, start_iteration+iteration_num):
    print(f"Begin the {t}th training round......\n") 
    
    model_index = t
    path = 'server/W_'+str(model_index)+'_th.h5'  

    # local training
    for i in range(user_num):
        
        # Step(1) initialize local model and load data
        local_model = load_model(path)
        x_train, y_train, x_test, y_test, y_true = load_client_data(i) 
        
        # Step(2) observe and record the state
        loss, acc, fs = observe(local_model, x_train, y_train, y_true)
        local_loss[i].append(loss)
        local_acc[i].append(acc)
        local_fs[i].append(fs)
    
        # Step(3) choose an action
        if t == 0: 
            # randomly select the intial action in the 0-th training round
            action = [0, 0]
            action[0] = uniform(-1, 1)            
            action[1] = uniform(-1, 1)
        else:
            # else, first, restore ddpg model
            ddpg.restore_model("users/user_"+str(i)+"/local_ddpg_model/ddpg")
            try: # if exists, load menmory to speed up training
                ddpg.memory = np.load('ddpg_train_samples2.npz')['arr_0'][:1000]
            except:
                pass
            
            # then, compute and record the reward
            phi1, phi2, phi3 = local_loss[i][t-1]-local_loss[i][t], local_acc[i][t]-local_acc[i][t-1], local_fs[i][t]-local_fs[i][t-1]
            r = xi1*phi1+ xi2*phi2 + xi3*phi3
            reward[i].append(r)
            print(f"The reward of client {i} in the {t}th training round is {r}")
            
            # then, simulate the training and transmission cost
            c = uniform(-1, 1)
            cost[i].append(c)
            
            # then, form and store the trans, i.e., [s(t-1), a(t-1), r(t-1), c(t), s(t)]
            trans = [[local_loss[i][t-1], local_acc[i][t-1], local_fs[i][t-1], \
                     local_action0[i][t-1], local_action1[i][t-1], \
                     reward[i][t-1], \
                     cost[i][t-1], \
                     local_loss[i][t], local_acc[i][t], local_fs[i][t]]]
            ddpg.memory = np.concatenate((ddpg.memory, trans), axis=0)
            
            # then, update local DDPG mode
            actor_loss_value, critic_loss_value, cost_loss_value = ddpg.learn()
            
            # then, record ddpg loss and save local DDPG mode
            actor_loss[i].append(actor_loss_value)
            critic_loss[i].append(critic_loss_value)
            cost_loss[i].append(cost_loss_value)
            # ddpg.save_model("users/user_"+str(i)+"/local_ddpg_model/ddpg")
            print(f"The client {i}'s local ddpg model is trained and saved")
        
            # then, call ddpg to output an action
            state = np.array([loss, acc, fs])
            action = ddpg.choose_action(state)
        
        # Step(4) compute training hypermaters according to action
        # shift = 0 if the last dense of actor model doesn't have activation
        second_learning_rate = pow(base_learning_rate, action[0]-3) # shift -3
        second_train_epoch_num = int(base_iteration*(action[1]+1)) # shift 1
        
        # # set constant training hypermaters if needed, i.e., ddpg-eta and ddpg-alpha
        # second_learning_rate = 1e-3
        # second_train_epoch_num = 15
        
        # Step(5) record action and hypermaters
        local_action0[i].append(action[0])
        local_action1[i].append(action[1])
        learning_rate[i].append(second_learning_rate)
        epoch_num[i].append(second_train_epoch_num)
        print(f"The learning rate and training epoch of client {i} is {second_learning_rate} and {second_train_epoch_num}")
        
        # Step(6) excute local training and save updated local model
        local_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(second_learning_rate), metrics=['accuracy'])
        local_model.fit(x_train, y_train, batch_size=32, epochs=second_train_epoch_num, verbose=0) 
        local_model.save('users/user_'+str(i)+'/'+str(t+1)+'_th.h5')
        print(f"Finish second training for client {i} in the {t}th training round\n") 
    
    # Step(7) global aggregation
    loss, acc, fs = average_model(user_num, weights, t, "cnn") # CNN on MNIST / Fashion-MNIST / FEMNIST
    # loss, acc, fs = average_model(user_num, weights, t, "logistic") # Logistic on MNIST
    # loss, acc, fs = average_model(user_num, weights, t, "resnet") # ResNet-18 on MNIST
    global_loss.append(loss)
    global_acc.append(acc)
    global_fs.append(fs)
    print(f"The performence of global model is, loss: {loss}; acc: {acc}; fs: {fs}\n")

# save training statues for each clients
for i in range(user_num):
    save_csv(local_loss[i], "users/user_"+str(i)+"/local_loss.csv")
    save_csv(local_acc[i], "users/user_"+str(i)+"/local_acc.csv")
    save_csv(local_fs[i], "users/user_"+str(i)+"/local_fs.csv")
    save_csv(learning_rate[i], "users/user_"+str(i)+"/learning_rate.csv")
    save_csv(epoch_num[i], "users/user_"+str(i)+"/epoch_num.csv")
    save_csv(reward[i], "users/user_"+str(i)+"/reward.csv")
    save_csv(cost[i], "users/user_"+str(i)+"/cost.csv")
    save_csv(actor_loss[i], "users/user_"+str(i)+"/actor_loss.csv")
    save_csv(critic_loss[i], "users/user_"+str(i)+"/critic_loss.csv")

# save training statues for server
save_csv(global_loss, "server/global_loss.csv")
save_csv(global_acc, "server/global_acc.csv")
save_csv(global_fs, "server/global_fs.csv")
