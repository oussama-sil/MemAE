import tensorflow as tf 
import numpy as np 



# Compute the reconstruction error between two frames 
def reconstruct_error(x_rec,x_ori):
    """
        Function  to compute the reconstruction error 
        Input x_rec,x_ori of shape (m,T,H,W,1)
    """
    tmp = (x_rec-x_ori)**2
    tmp = tf.reduce_sum(tmp, axis=[-1, -2,-3,-4,-5])
    return  tmp



def EntropyLoss(eps=1e-12):
    def entropy_loss(w):
        """
            Function  the entropy loss of the weights 
            Input wi of shape (m,T,H,W,N)
        """
        b = -w*tf.math.log(w+eps)
        b =  tf.math.reduce_sum(b,axis=-1) # sum over axe of N
        return tf.reduce_mean(b)
    return entropy_loss


#? Gradient loss for the blur in images 
def grad_loss(x_rec,x_ori):
    """
        Input:
            x_ori,x_rec : reconstructed and original image of shape (m,T,H,W,1)
        Output :
            gradient loss between reconstructed and originale image 
    """

    #? Casting type 


    x_rec = reshaped_tensor = tf.reshape(x_rec, (-1, x_rec.shape[2], x_rec.shape[3], x_rec.shape[4]))
    x_ori = reshaped_tensor = tf.reshape(x_ori, (-1, x_ori.shape[2], x_ori.shape[3], x_ori.shape[4]))

    grad_y_rec,grad_x_rec = tf.math.abs(tf.image.image_gradients(x_rec))
    grad_y_ori,grad_x_ori = tf.math.abs(tf.image.image_gradients(x_ori))
        
    grad_y_rec = tf.cast(grad_y_rec, dtype=tf.float32)
    grad_x_rec = tf.cast(grad_x_rec, dtype=tf.float32)
    grad_y_ori = tf.cast(grad_y_ori, dtype=tf.float32)
    grad_x_ori = tf.cast(grad_x_ori, dtype=tf.float32)
    
    grad_diff_y = tf.math.abs(grad_y_ori - grad_y_rec)
    grad_diff_x = tf.math.abs(grad_x_ori - grad_x_rec)

    return tf.reduce_mean(grad_diff_x + grad_diff_y) 




def MemAELoss(alpha=0.0002,eps=1e-12,alpha_grad_loss=0.001):
    """
        Input : 
            alpha : Entropy Loss Weight
            eps : epsilon to add to the weights before computing the log (should be small)
        Output :
            MemAE model loss function (MSE + alpha*EntropyLOss)
    """
    reconstruct_error_fct = tf.keras.losses.MeanSquaredError(reduction= 'sum_over_batch_size')
    entropy_loss = EntropyLoss(eps=eps)
    def memae_loss(x_rec,x_ori,w):
        print(reconstruct_error_fct(x_ori, x_rec))
        print(grad_loss(x_ori, x_rec))
        print(entropy_loss(w))
        # print(w[0,0,0,0,0])
        return reconstruct_error_fct(x_ori, x_rec)+alpha*entropy_loss(w) +alpha_grad_loss*entropy_loss(w)
    return memae_loss

if __name__=="__main__":
    input_shape =  (2, 2, 4, 4, 1)

    x_rec = tf.random.normal(input_shape)
    x_ori = tf.random.normal(input_shape)

    print(x_rec[0,0,:,:,0])

    x_rec = reshaped_tensor = tf.reshape(x_rec, (-1, x_rec.shape[2], x_rec.shape[3], x_rec.shape[4]))
    x_ori = reshaped_tensor = tf.reshape(x_ori, (-1, x_ori.shape[2], x_ori.shape[3], x_ori.shape[4]))

    grad_y_rec,grad_x_rec = tf.math.abs(tf.image.image_gradients(x_rec))
    grad_y_ori,grad_x_ori = tf.math.abs(tf.image.image_gradients(x_ori))
    
    grad_diff_y = tf.math.abs(grad_y_ori - grad_y_rec)
    grad_diff_x = tf.math.abs(grad_x_ori - grad_x_rec)

    print(tf.reduce_mean(grad_diff_x + grad_diff_y))

