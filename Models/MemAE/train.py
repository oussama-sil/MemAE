import tensorflow as tf 
import numpy as np
import time 
import sys 
import os
import json
import pickle
from Models.MemAE.loss import MemAELoss


###########################################################################
##* MemAE trainning functions 
###########################################################################



def memae_apply_gradient(optimizer, model,loss_function, x):
    """
        Apply the gradient on the model's weights in memae model
        Inputs : 
            x : A batch of data
            loss_function : the loss function 
            model : model to train
            optimizer : optimizer to use in the update of the weights (Adam...) 
        Outputs :
            loss_value : computed loss value 
    """
    with tf.GradientTape() as tape:
        x_hat,w_hat = model(x)
        # print(x_hat.shape)
        loss_value = loss_function(x_hat, x,w_hat)
        # print(loss_value) 
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return  loss_value


# Loop over all the batches and perform one epoch 
def memae_train_data_for_one_epoch(train_data,optimizer, model,loss_function,validation_data,evaluation_metric):
    """
        Train the model for one epoch 
        Inputs : 
            train_data : Data to train on
            loss_function : the loss function 
            model : model to train
            optimizer : optimizer to use in the update of the weights (Adam...) 
            epoch_id : the current epoch 
            validation_data : data to use in the validation 
            evaluation_metric : function to evaluate the model
        Outputs :
            train_loss : loss on train data
            val_loss : loss on validation data
            eval_metric : evaluation metric value 
    """

    losses = [] 
    # Loop over trainning batches 
    for batch in range(train_data.shape[0]):
        # Perform backward on one batch 
        loss_value = memae_apply_gradient(optimizer, model,loss_function, train_data[batch,:,:,:,:,:]) 
        print(f"Btach {batch}/{train_data.shape[0]}  loss= {loss_value}")
        #Append loss for the current batch 
        losses.append(loss_value)

        # print('.')
    train_loss = np.mean(np.array(losses))
    val_loss = 0
    eval_metric = 0
    # Evaluation of the model on the validation data
    if len(validation_data) > 0 :
        validation_data_hat,validation_w_hat = model(validation_data)
        val_loss = loss_function(validation_data_hat, validation_data,validation_w_hat)
        if evaluation_metric != None :
            eval_metric = evaluation_metric(validation_data_hat,validation_data)


    #! When trainning in the graph mode 
    # print(f"Trainning loss after {epoch_id} epoch  = { tf.reduce_mean(tf.stack(losses), axis=0).eval()} ")
    # tf.print(tf.reduce_mean(tf.stack(losses), axis=0))
    return train_loss,val_loss,eval_metric


def train_memae(model,train_data,optimizer,validation_data=[],evaluation_metric=None,epochs=10,batch_size=16,loss_alpha=0.0002,loss_eps=1e-12,
            record_history = False,history_params= None):

    """
        Train the model for one epoch 
        Inputs : 
            train_data : Data to train on
            loss_function : the loss function 
            model : model to train
            optimizer : optimizer to use in the update of the weights (Adam...) 
            epochs : the number of epochs 
            batch_size 
        Outputs :
            train_loss : loss on train data
            val_loss : loss on validation data
            eval_metric : evaluation metric value 
    """

    nb_batches = int(train_data.shape[0] / batch_size)


    print("\n<<<  Trainning   >>>")
    print(f"Shape of data : {train_data.shape}")
    print(f"Size of a batch : {batch_size}   Number of batchs = {nb_batches}")
    print(f"Number of epochs : {epochs}")

    
    #? Losses and evaluation metrics 
    train_losses,val_losses,eval_metrics = [],[],[]

    #? Loss function for the model MemAE
    loss_function = MemAELoss(loss_alpha,loss_eps)


    #? Record of history
    if record_history and history_params != None:
        #? Create the folder 
        if not os.path.exists(history_params["dir"]):
            # Create the folder
            os.makedirs(history_params["dir"])
        
        #? Write the file that contains the parameters of the model and the trainning
        with open(os.path.join(history_params["dir"], "params.json"), "w") as file:
            # Write the dictionary to the file
            json.dump(history_params["train_params"], file)

        #? Create directory for the weights checkpoint  
        os.makedirs(os.path.join(history_params["dir"], "Weights"))

        #? Create the file for the trainning time 
        with open(os.path.join(history_params["dir"], "summary.txt"), "w") as file:
            file.write("Results : \n")

        #? Create file for the evolution of the loss 
        with open(os.path.join(history_params["dir"], "loss.txt"), "w") as file:
            file.write("Train_Loss,Test_Loss,Eval_Metric\n")
    
    overall_train_time = time.time()
    # Iterate over epochs.
    for epoch in range(epochs):
        
        #? Making the trainning batchs for the epoch 
        # Shuffling the trainning data
        indices = np.arange(train_data.shape[0]) 
        np.random.shuffle(indices)
        train_data = train_data[indices]

        # Split the input array into multiple batches
        batches = [train_data[i*batch_size:i*batch_size+batch_size,:,:,:,:] for i in range(nb_batches)]

        #TODO:  Freeing the array train_data ==> To add        

        print(f'\nStart of epoch {epoch+1} ' )

        start_time = time.time()
        train_loss,val_loss,eval_metric = memae_train_data_for_one_epoch(np.array(batches,dtype=np.float32),optimizer, model,loss_function,validation_data,evaluation_metric)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        eval_metrics.append(eval_metric)

        print(f"Trainning loss after {epoch+1} epoch  = { train_loss } , on validation data {val_loss}, Eval metric {eval_metric}")
        print(f'End of epoch {epoch+1}, elapsed time = {round(time.time() - start_time,2)} (s)\n')

        #? Record of history
        if record_history and history_params != None:
            
            #? Saving the epoch results 
            with open(os.path.join(history_params["dir"], "summary.txt"), "a") as file:
                file.write(f"Trainning loss after {epoch+1} epoch  = { train_loss } , on validation data {val_loss}, Eval metric {eval_metric}\n")

            #? Saving the loss and the evaluation metric 
            with open(os.path.join(history_params["dir"], "loss.txt"), "a") as file:
                file.write(f"{train_loss},{val_loss},{eval_metric}\n")

            print("History recorded")
            #? Saving the weights and the optimizer state 
            if epoch % history_params["nb_epochs_checkpoint"] == 0:
                model.save_weights(os.path.join(history_params["dir"], f"Weights/weights_epoch_{epoch+1}.h5"))
                print("Weights Saved")

                with open(os.path.join(history_params["dir"], f"Weights/optimizer_epoch_{epoch+1}.pkl"), "wb") as file:
                    pickle.dump(optimizer.get_config(), file)

                print("Optimizer state Saved")

    print('\n')
    print(f'End of trainning, trainning time = {round(time.time() - overall_train_time,2)} (s)\n')


    return train_losses,val_losses,eval_metrics







###########################################################################



def apply_gradient(optimizer, model,loss_function, x):
    """
        Apply the gradient on the model's weights 
        Inputs : 
            x : A batch of data
            loss_function : the loss function 
            model : model to train
            optimizer : optimizer to use in the update of the weights (Adam...) 
        Outputs :
            loss_value : computed loss value 
    """
    with tf.GradientTape() as tape:
        x_hat = model(x)
        # print(x_hat.shape)
        loss_value = loss_function(x_hat, x)
        # print(loss_value) 
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return  loss_value


# Loop over all the batches and perform one epoch 
def train_data_for_one_epoch(train_data,optimizer, model,loss_function,validation_data,evaluation_metric):
    """
        Train the model for one epoch 
        Inputs : 
            train_data : Data to train on
            loss_function : the loss function 
            model : model to train
            optimizer : optimizer to use in the update of the weights (Adam...) 
            epoch_id : the current epoch 
            validation_data : data to use in the validation 
            evaluation_metric : function to evaluate the model
        Outputs :
            train_loss : loss on train data
            val_loss : loss on validation data
            eval_metric : evaluation metric value 
    """

    losses = [] 
    # Loop over trainning batches 
    for batch in range(train_data.shape[0]):
        # Perform backward on one batch 
        loss_value = apply_gradient(optimizer, model,loss_function, train_data[batch,:,:,:,:,:]) 
        
        #Append loss for the current batch 
        losses.append(loss_value)

        print('.')
    train_loss = np.mean(np.array(losses))
    val_loss = 0
    eval_metric = 0
    # Evaluation of the model on the validation data
    if len(validation_data) > 0 :
        validation_data_hat = model(validation_data)
        val_loss = loss_function(validation_data_hat, validation_data)
        if evaluation_metric != None :
            eval_metric = evaluation_metric(validation_data_hat,validation_data)


    #! WHen trainning in the graph mode 
    # print(f"Trainning loss after {epoch_id} epoch  = { tf.reduce_mean(tf.stack(losses), axis=0).eval()} ")
    # tf.print(tf.reduce_mean(tf.stack(losses), axis=0))
    return train_loss,val_loss,eval_metric


def train_ae(model,train_data,optimizer,loss_function,validation_data=[],evaluation_metric=None,epochs=10,batch_size=16,
             record_history = False,history_param= None):

    """
        Train the model for one epoch 
        Inputs : 
            train_data : Data to train on
            loss_function : the loss function 
            model : model to train
            optimizer : optimizer to use in the update of the weights (Adam...) 
            epochs : the number of epochs 
            batch_size 
        Outputs :
            train_loss : loss on train data
            val_loss : loss on validation data
            eval_metric : evaluation metric value 
    """

    nb_batches = int(train_data.shape[0] / batch_size)


    print("\n<<<  Trainning   >>>")
    print(f"Shape of data : {train_data.shape}")
    print(f"Size of a batch : {batch_size}   Number of batchs = {nb_batches}")
    print(f"Number of epochs : {epochs}")

    
    #? Losses and evaluation metrics 
    train_losses,val_losses,eval_metrics = [],[],[]

    # Iterate over epochs.
    for epoch in range(epochs):
        
        # Shuffling the trainning data
        indices = np.arange(train_data.shape[0]) 
        np.random.shuffle(indices)
        train_data = train_data[indices]

        # Split the input array into multiple batches
        batches = [train_data[i*batch_size:i*batch_size+batch_size,:,:,:,:] for i in range(nb_batches)]

        #TODO:  Freeing the array train_data ==> To add        

        print(f'\nStart of epoch {epoch+1} ' )

        start_time = time.time()
        train_loss,val_loss,eval_metric = train_data_for_one_epoch(np.array(batches,dtype=np.float32),optimizer, model,loss_function,validation_data,evaluation_metric)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        eval_metrics.append(eval_metric)

        print(f"Trainning loss after {epoch} epoch  = { train_loss } , on validation data {val_loss}, Eval metric {eval_metric}")
        print(f'End of epoch {epoch+1}, elapsed time = {round(time.time() - start_time,2)} (s)\n')

    print('\n')



    return train_losses,val_losses,eval_metrics



#? Trainning a model on a pipelined dataset 
@tf.function
def train_ae_td_dataset(model,train_data,optimizer,loss_function,epochs=10,batch_size=16):

    nb_batches = int(train_data.shape[0] / batch_size)


    print("\n<<<  Trainning   >>>")
    print(f"Shape of data : {train_data.shape}")
    print(f"Size of a batch : {batch_size}   Number of batchs = {nb_batches}")
    print(f"Number of epochs : {epochs}")

    
    # Iterate over epochs.
    for epoch in range(epochs):
        
        # Shuffling the trainning data
        indices = np.arange(train_data.shape[0]) # Size of the dataset 
        np.random.shuffle(indices)
        train_data = train_data[indices]

        # Split the input array into multiple batches
        batches = [train_data[i*batch_size:i*batch_size+batch_size,:,:,:,:] for i in range(nb_batches)]

        #TODO:  Freeing the array train_data ==> To add        

        print(f'\nStart of epoch {epoch+1} ' )
        start_time = time.time()
        losses_train = train_data_for_one_epoch(np.array(batches,dtype=np.float32),optimizer, model,loss_function,epoch)
        print(f'End of epoch {epoch+1}, elapsed time = {round(time.time() - start_time,2)} (s)\n')
