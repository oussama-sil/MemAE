import tensorflow as tf 
import numpy as np 




def reconstruct_error_sequence(x_rec,x_ori):
    """
        Compute the reconstruction error for a batch of sequences 
        Input :
            x_rec : Reconstructed sequences of shape (m,T,H,W,1)
            x_ori : Original sequence of shape  (m,T,H,W,1)
        Output :
            rec_error : Reconstruction erro of each 
    """

    rec_error = (x_rec-x_ori)**2
    rec_error = tf.reduce_sum(rec_error, axis=[-1,-2,-3])
    return rec_error 


def NormalityScore(x_rec,x_ori):
    """
        Compute the normality score for each frame in a batch of sequences 
        Input :
            x_rec : Reconstructed sequences of shape (m,T,H,W,1)
            x_ori : Original sequence of shape  (m,T,H,W,1)
        Output :
            Normality  : Reconstruction erro of each 
    """
    rec_error = reconstruct_error_sequence(x_rec,x_ori)
    max = tf.reduce_max(rec_error,axis=[-1])
    min = tf.reduce_min(rec_error,axis=[-1])

    return 1 - (rec_error-min) / (max - min)


if __name__=='__main__':

    input_shape = (6,8,128,128,1)
    x_ori = np.ones(input_shape) 
    x_ori = random_array = np.random.random(input_shape)
    x_rec = np.ones(input_shape)*0.5


    rec_error = reconstruct_error_sequence(x_rec,x_ori)
    max = tf.reduce_max(rec_error,axis=[-1],keepdims=True)
    min = tf.reduce_min(rec_error,axis=[-1],keepdims=True)

    # norm_score = 1 - (rec_error-min) / (max - min)

    print(min)
    print(rec_error[0,0])
