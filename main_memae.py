import time 
import numpy as np
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # To disable trainning in GPU
import tensorflow as tf 
tf.get_logger().setLevel(tf._logging.ERROR)

# Data Loader 
from Data.data_loader import create_train_dataset_memae  # Trainning dataset loader for MemAE model 
from Data.data_loader import create_test_dataset_memae  # Trainning dataset loader for MemAE model 


# Importing the models 
from Models.MemAE.model import AE
from Models.MemAE.model import MemAE

# Importing trainning function 
# from Models.MemAE.train import train_ae
from Models.MemAE.train import train_memae

# Loss function 
# from  Models.MemAE.loss import reconstruct_error
# from  Models.MemAE.loss import MemAELoss

# Visualization functions 
from Utils.visualization import plot_two_images,plot_result
from Utils.plots import plot_loss_evol,plot_metric_evol

#? Loading trainning dataset for MemAE model 
data_train = create_train_dataset_memae(64,64,16,load_ped2 = True ,load_ped1=False,shuffle=False,m=154,save=True,save_path='./Data',path_ped2='./Data/UCSDped2',path_ped1='./Data/UCSDped1')
#? Loading the test data 
data_test,data_test_gt,data_test_labels = create_test_dataset_memae(64,64,16,load_ped2 = True ,load_ped1=False,save=True,save_path='./Data',path_ped2='./Data/UCSDped2',path_ped1='./Data/UCSDped1')

#? Creation of the trainning and testing pipeline 
# batch_size = 32
# train_dataset = tf.data.Dataset.from_tensor_slices(data_train)
# # del(data_train)
# # train_dataset  = train_dataset.shuffle(buffer_size=10,reshuffle_each_iteration=None,)
# # train_dataset  = train_dataset.batch(50, drop_remainder=True)

# train_dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
# del(data_train)
# train_dataset  = train_dataset.shuffle(buffer_size=5,reshuffle_each_iteration=True)

# train_dataset  = train_dataset.batch(5, drop_remainder=True,num_parallel_calls=6)
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))

# train_dataset  = train_dataset.batch(5, drop_remainder=True,num_parallel_calls=5)
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))
# print(next(iter(train_dataset)))

# # Define the number of CPU cores to utilize
# num_cpu_cores = 4

# # Configure TensorFlow to use parallelism
# config = tf.compat.v1.ConfigProto(
#     intra_op_parallelism_threads=num_cpu_cores,
#     inter_op_parallelism_threads=num_cpu_cores,
# )
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

# # initialize_all_variables().run() 


# with tf.compat.v1.Session() as sess:

tf.config.threading.set_inter_op_parallelism_threads(6) 
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.set_soft_device_placement(False)


if True: 

    #? Parameters of the model 

    learning_rate = 0.0002 #0.0001 gave good results 
    nb_epochs = 2
    mem_dim = 2000
    entropy_loss = 0.002 # 0.002 
    shrink_thresh = 0.0005 #0.0025  0.001 0.0015 0.0005 0.0004
    batch_size = 6
    #? Model

    memae_model = MemAE((None,16,64,64,1),nb_items=mem_dim,
                        delta=shrink_thresh)
    memae_model.build((None,16,64,64,1))
    memae_model.summary()

    #? History params 
    history_params = {
        "dir" : "./Trainning_History/MemAE", # Folder in which save the data  
        "train_params" : {
            "train_data_shape": data_train.shape,
            "test_data_shape" : data_test.shape,
            "learning_rate" : learning_rate, 
            "nb_epochs" : nb_epochs,
            "mem_dim" :mem_dim,
            "entropy_loss" : entropy_loss, 
            "shrink_thresh" :shrink_thresh,
            "batch_size":batch_size
        },
        "nb_epochs_checkpoint" : 1,  # Save each nb_epochs_checkpoint epoch 
    }





    #? Optimizer 
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #? To load the optimizer state 
    # with open("./Trainning_History/MemAE/Weights/optimizer_epoch_2.pkl", "rb") as file:
    #     optimizer_config = pickle.load(file)
    #     optimizer =tf.keras.optimizers.Adam().from_config(optimizer_config)



    TRAIN = True

    if TRAIN:
        
        # memae_model.load_weights("./Models/memae_model_weights_(128,128).h5")

        #? Train model 
        train_losses,val_losses,eval_metrics = train_memae(memae_model,data_train,optimizer,
                validation_data=data_test[5:10,:,:,:,:],
                evaluation_metric=None,epochs=nb_epochs,batch_size=batch_size,
                record_history=True,history_params=history_params)

        memae_model.save_weights("./Models/memae_model_weights_(256,256).h5")


        plot_loss_evol(train_losses,val_losses)

    else:
        #? Loading a model 
        # memae_model.load_weights("./Models/memae_model_weights_(128,128)_lr_0.001.h5")
        memae_model.load_weights("./Models/memae_model_weights_(256,256).h5")
        
        # ae_model.summary()


    # Testing the model with images from the trainning dataset or the test_dataset
    in_img = data_test[8:9,:,:,:,:]
    out_img = memae_model(in_img)[0]
    rec_error = (in_img-out_img)**2
    anomaly_mask = np.where(rec_error.numpy() > 0.05 , 1, 0)

    # plot_two_images(in_img[0,2,:,:,0],rec_error[0,2,:,:,0])
    plot_result(in_img[0,-1,:,:,:],out_img[0,-1,:,:,:],rec_error[0,-1,:,:,:],anomaly_mask[0,-1,:,:,:],save=True,filename='/media/osil/Secondary/_Stage_CERIST/Documentation_Paper/images/MemAE/ae_results.png')
