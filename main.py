import time 
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # To disable trainning in GPU
import tensorflow as tf 

# Data Loader 
from Data.data_loader import create_train_dataset_memae  # Trainning dataset loader for MemAE model 
from Data.data_loader import create_test_dataset_memae  # Trainning dataset loader for MemAE model 


# Importing the models 
from Models.MemAE.model import AE

# Importing trainning function 
from Models.MemAE.train import train_ae

# Loss function 
from  Models.MemAE.loss import reconstruct_error

# Visualization functions 
from Utils.visualization import plot_two_images,plot_result
from Utils.plots import plot_loss_evol,plot_metric_evol

#? Loading trainning dataset for MemAE model 
data_train = create_train_dataset_memae(128,128,16,load_ped2 = True ,load_ped1=False,shuffle=False,m=154,save=True,save_path='./Data',path_ped2='./Data/UCSDped2',path_ped1='./Data/UCSDped1')
#? Loading the test data 
data_test,data_test_gt,data_test_labels = create_test_dataset_memae(128,128,16,load_ped2 = True ,load_ped1=False,save=True,save_path='./Data',path_ped2='./Data/UCSDped2',path_ped1='./Data/UCSDped1')

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

tf.config.threading.set_inter_op_parallelism_threads(4) 
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.set_soft_device_placement(False)


if True:

    #? Creation of the model 
    #* Model with simple encoder decoder 
    ae_model = AE((None,16,128,128,1))
    ae_model.build((None,16,128,128,1))
    ae_model.summary()
    # time.sleep(30)

    #? Optimizer 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)



    TRAIN = True

    if TRAIN:

        #? Trainning the model 
        ae_model.load_weights("./Models/ae_model_weights_(128,128).h5")
        # train_ae(ae_model,data_train,optimizer,reconstruct_error,10)
        train_losses,val_losses,eval_metrics = train_ae(ae_model,data_train,optimizer,tf.keras.losses.MeanSquaredError(reduction= 'sum_over_batch_size'),
                                                        validation_data=data_test[5:10,:,:,:,:],evaluation_metric=None,epochs=0,batch_size=16)

        plot_loss_evol(train_losses,val_losses)

        #? Saving the trained model weights 
        # ae_model.save_weights("./Models/ae_model_weights_(128,128).h5")
        
    else:
        #? Loading a model 
        ae_model.load_weights("./Models/ae_model_weights_(128,128).h5")
        # ae_model = tf.keras.models.load_model('./Models/ae_(64,64)_2')
        # ae_model.summary()


    # Testing the model with images from the trainning dataset or the test_dataset
    in_img = data_test[8:9,:,:,:,:]
    out_img = ae_model(in_img)
    rec_error = (in_img-out_img)**2
    anomaly_mask = np.where(rec_error.numpy() > 0.05 , 1, 0)

    # plot_two_images(in_img[0,2,:,:,0],rec_error[0,2,:,:,0])
    plot_result(in_img[0,-1,:,:,:],out_img[0,-1,:,:,:],rec_error[0,-1,:,:,:],anomaly_mask[0,-1,:,:,:],save=True,filename='/media/osil/Secondary/_Stage_CERIST/Documentation_Paper/images/MemAE/ae_results.png')
