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
            Input x_rec,x_ori of shape (m,T,H,W,1)
        """
        b = -w*tf.math.log(w+eps)
        return tf.reduce_mean(b)
    return entropy_loss


def MemAELoss(alpha=0.0002,eps=1e-12):
    """
        Input : 
            alpha : Entropy Loss Weight
            eps : epsilon to add to the weights before computing the log (should be small)
    """
    reconstruct_error_fct = tf.keras.losses.MeanSquaredError(reduction= 'sum_over_batch_size')
    entropy_loss = EntropyLoss(eps=eps)
    def memae_loss(x_rec,x_ori,w):
        # print(reconstruct_error_fct(x_ori, x_rec))
        # print(entropy_loss(w))
        # print(w[0,0,0,0,0])
        return reconstruct_error_fct(x_ori, x_rec)+alpha*entropy_loss(w)
    return memae_loss

if __name__=="__main__":
    t1 = [[[1,2],[3,4]],[[1,2],[3,4]]]
    t1 = np.array(t1)
    t1 = t1.reshape((1,2,2,2))
    t2 = [[[3,2],[3,4]],[[1,1],[3,4]]]
    t2 = np.array(t2)
    t2 = t2.reshape((1,2,2,2))
    print(t1)
    print(t2)

    # print(tf.norm(tf.constant(t1-t2,dtype=tf.float32)))
    
    print(reconstruct_error(t1,t2))

