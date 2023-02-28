import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model


from sklearn.metrics import f1_score
from numpy import argmax
from copy import deepcopy
from build_temp_model import build_cnn_model as cnn
from build_temp_model import build_logistic_model as log
from build_temp_model import build_resnet_model as res

import warnings
warnings.filterwarnings("ignore")

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



def load_server_test_data():
    # modify here to load other test data
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
    updated_global_model = load_model('users/user_1'+'/'+str(t+1)+'_th.h5')
    updated_global_model.set_weights(temp_model_weight)
    updated_global_model.save('server/W_'+str(t+1)+'_th.h5')
    
    # server tests global model
    server_x_test, server_y_test, server_y_true = load_server_test_data()
    loss, acc, fs = observe(local_model, server_x_test, server_y_test, server_y_true)
    return loss, acc, fs


def observe(model, x_test, y_test, y_true):
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    fs = compute_F1_socre(model, x_test, y_true)
    return loss, acc, fs


def save_csv(filedata, filepath):
    temp=pd.DataFrame(filedata)
    temp.to_csv(filepath, encoding='gbk', header=0, index=None) 


# define parameters
user_num = 10 # the number of clients
start_iteration = 0 # the start of local training
iteration_num = 5 # the number of training rounds sustained in local training
second_batch_size = 32 # the batch size for local training
# small setting
second_learning_rate = 1e-4
second_train_epoch_num = 1
# # large setting
# second_learning_rate = 1e-2
# second_train_epoch_num = 25

# define the recorders
empty_recoder = [[] for _ in range(user_num)]
local_loss, local_acc, local_fs = deepcopy(empty_recoder), deepcopy(empty_recoder), deepcopy(empty_recoder) # local state record
global_loss, global_acc, global_fs = [], [], [] # recorded by server

# define the aggregation weight, i.e., the size of lacal data, average 600
weights = [600, 600, 600, 600, 600, 600, 600, 600, 600, 600]


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
    
        # Step(3) excute local training and save updated local model
        local_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(second_learning_rate), metrics=['accuracy'])
        local_model.fit(x_train, y_train, batch_size=32, epochs=second_train_epoch_num, verbose=0) 
        local_model.save('users/user_'+str(i)+'/'+str(t+1)+'_th.h5')
        print(f"Finish second training for client {i} in the {t}th training round\n") 
    
    # Step(4) global aggregation
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
    
# save training statues for server
save_csv(global_loss, "server/global_loss.csv")
save_csv(global_acc, "server/global_acc.csv")
save_csv(global_fs, "server/global_fs.csv")