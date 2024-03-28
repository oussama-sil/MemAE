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

# Visualization functions 
from Utils.visualization import plot_two_images,plot_result
from Utils.plots import plot_loss_evol,plot_metric_evol

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

tf.config.threading.set_inter_op_parallelism_threads(12) 
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.set_soft_device_placement(False)


# ? Loading data 
#* Dimensions of data 
#* PED2 are 3:2
H = 128 # 192 160
W = 192 # 128  224
nb_frames = 16
#* Loading trainning dataset for MemAE model 
data_train = create_train_dataset_memae(H,W,nb_frames,load_ped2 = True ,load_ped1=False,shuffle=True,m=350,save=True,save_path='./Data/Data_npy',path_ped2='./Data/UCSDped2',path_ped1='./Data/UCSDped1')
#* Loading the test data 
data_test,data_test_gt,data_test_labels = create_test_dataset_memae(H,W,nb_frames,load_ped2 = True ,load_ped1=False,save=True,save_path='./Data/Data_npy',path_ped2='./Data/UCSDped2',path_ped1='./Data/UCSDped1')


#? Parameters of the model 
#TODO : try with 0.0025 (same params as paper and github and with other optimizer)
#TODO : look for other method
#TODO : implement normality score
#TODO : implement AUC
learning_rate = 0.0001 #0.0001/0.0002 gave good results 
nb_epochs = 120
mem_dim = 2000
entropy_loss = 0.0002  # 0.002 goood res , 0.002  / 0.0002
shrink_thresh = 0.0025 #0.0025  0.001 0.0015 0.0005 0.0004
batch_size = 14
alpha_grad_loss = 0 # Grad loss 

#? History params 
history_params = {
            "dir" : f"./Training_History/MemAE_{H}_{W}", # Folder in which save the data  
            "train_params" : {
                "train_data_shape": data_train.shape,
                "test_data_shape" : data_test.shape,
                "learning_rate" : learning_rate, 
                "nb_epochs" : nb_epochs,
                "mem_dim" :mem_dim,
                "entropy_loss" : entropy_loss, 
                "shrink_thresh" :shrink_thresh,
                "batch_size":batch_size,
                "alpha_grad_loss":alpha_grad_loss
            },
            "nb_epochs_checkpoint" : 10,  # Checkpoint (saving weights and optimizer state) each .. epochs 
        }



#? Model
memae_model = MemAE((None,nb_frames,H,W,1),nb_items=mem_dim,
                            delta=shrink_thresh)
memae_model.build((None,nb_frames,H,W,1))
memae_model.summary()


#? Train or load pre trained  ?
TRAIN = True

if TRAIN: 

    #? Creation of a new optimizer 
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) 
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9) 
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    #optimizer = tf.keras.optimizers.Adagrad() 
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    history_params["train_params"]["optimizer_name"] = optimizer.get_config()["name"]
    # history_params["train_params"]["optimizer_learning_rate"] = optimizer.get_config()["learning_rate"]

    # print(history_params["train_params"])
    #TODO : Try other optimizer 
    #? Loading an old optimizer 
    # with open("./Trainning_History/MemAE/Weights/optimizer_epoch_2.pkl", "rb") as file:
    #     optimizer_config = pickle.load(file)
    #     optimizer =tf.keras.optimizers.Adam().from_config(optimizer_config)

    # memae_model.load_weights("./Models/memae_model_weights_(256,256).h5")

    train_losses,val_losses,eval_metrics = train_memae(memae_model,data_train,optimizer,
                    validation_data=data_test[5:7,:,:,:,:],loss_alpha=entropy_loss,alpha_grad_loss=alpha_grad_loss,
                    evaluation_metric=None,epochs=nb_epochs,batch_size=batch_size,
                    record_history=True,history_params=history_params)

    # Saving the weights 
    memae_model.save_weights(f"./Models/Models_weights_save/memae_model_weights_({H},{W}).h5")


    plot_loss_evol(train_losses,val_losses)

else:


    #? Loading a model 
    memae_model.load_weights(f"./Models/Models_weights_save/memae_model_weights_({H},{W}).h5")
        
    memae_model.summary()


#? Testing the model with images from the trainning dataset or the test_dataset
in_img = data_test[14:15,:,:,:,:]
out_img,_ = memae_model(in_img)
rec_error = (in_img-out_img)**2 # Compute the reconstruction error 
anomaly_mask = np.where(rec_error.numpy() > 0.05 , 1, 0)

# plot_two_images(in_img[0,2,:,:,0],rec_error[0,2,:,:,0])
plot_result(in_img[0,-1,:,:,:],out_img[0,-1,:,:,:],rec_error[0,-1,:,:,:],anomaly_mask[0,-1,:,:,:],save=True,filename='/media/osil/Secondary/_Stage_CERIST/Documentation_Paper/images/MemAE/ae_results.png')
