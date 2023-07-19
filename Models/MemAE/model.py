import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Layer
import math 
# Building a simple encoder, decoder architecture model without the memore 

LEAKY_RELU = 0.2 # Leaky relu param


class Encoder(tf.keras.Model):
    def __init__(self, leaky_relu_param = LEAKY_RELU):
        super(Encoder, self).__init__(name='Encoder')
        
        # Input shape 16 256 256 1 


        # Parameters of the model 
        self.leaky_relu_alpha = leaky_relu_param

        # Layers of the model 
        # Bloc 01 
        self.Bloc1 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding3D(padding=1),
            tf.keras.layers.Conv3D(96,(3, 3, 3),(1, 2, 2),data_format="channels_last"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)])
        # Bloc 02
        self.Bloc2 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding3D(padding=1),
            tf.keras.layers.Conv3D(128,(3, 3, 3),(2,2,2),data_format="channels_last"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)])
        # Bloc 03
        self.Bloc3 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding3D(padding=1),
            tf.keras.layers.Conv3D(256,(3, 3, 3),(2,2,2),data_format="channels_last"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)])
        # Bloc 04
        self.Bloc4 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding3D(padding=1),
            tf.keras.layers.Conv3D(256,(3, 3, 3),(2, 2, 2),data_format="channels_last"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.leaky_relu_alpha),
        ])
    @tf.function
    def call(self, input_tensor):
        x = self.Bloc1(input_tensor)
        x = self.Bloc2(x)
        x = self.Bloc3(x)
        x = self.Bloc4(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(self, leaky_relu_param = LEAKY_RELU):
        super(Decoder, self).__init__(name='Decoder')

        # Input shape 2 16 16 256 

        # Parameters of the model 
        self.leaky_relu_alpha = leaky_relu_param


        # Layers of the model 
        # Bloc 01 
        self.Bloc1 = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(256,kernel_size	= (3, 3, 3),strides=(2, 2, 2),data_format="channels_last",padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)])
        # Bloc 02
        self.Bloc2 = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(128,kernel_size	= (3, 3, 3),strides=(2, 2, 2),data_format="channels_last",padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)])
        # Bloc 03
        self.Bloc3 = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(96,kernel_size	= (3, 3, 3),strides=(2, 2, 2),data_format="channels_last",padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)])
        # Bloc 04
        self.Bloc4 = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(1,kernel_size	= (3, 3, 3),strides=(1, 2, 2),data_format="channels_last",padding='same')
        ])
    @tf.function
    def call(self, input_tensor):
        x = self.Bloc1(input_tensor)
        x = self.Bloc2(x)
        x = self.Bloc3(x)
        x = self.Bloc4(x)
        return x


class AE(tf.keras.Model):

    def __init__(self,input_shape_dim=(None,16,128,128,1)):
        super(AE, self).__init__(name='AE')
        
        self.input_shape_dim = input_shape_dim

        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # print(self.input_shape_dim)
        # print((self.input_shape_dim[0],int(self.input_shape_dim[1]/8),int(self.input_shape_dim[2]/16),int(self.input_shape_dim[3]/16),self.input_shape_dim[4]))
        self.encoder.build(self.input_shape_dim)
        self.decoder.build((self.input_shape_dim[0],int(self.input_shape_dim[1]/8),int(self.input_shape_dim[2]/16),int(self.input_shape_dim[3]/16),256))
        
    

    @tf.function
    def call(self, input_tensor):
        z = self.encoder(input_tensor)
        x_hat = self.decoder(z)
        return x_hat


#? Memory Model

#* Layer 01 : SImilarity between the memory items and the input 

class Similarity(Layer):

    # To istanciate the layer and set the kernel parameters  
    def __init__(self,type="cosine",eps=1e-14):
        '''Initializes the instance attributes'''
        super(Similarity, self).__init__()
        self.type = type
        self.eps = eps
    # Perform the computation 
    def call(self, inputs):
        """
            Input
                z: of shape (m,2,H,W,C)
                m: of shape (N,C)
            Output 
                cosine similarity : of shape (m,2,H,W,N)
        """

        (z,m) = inputs 
        z_l2_norm = tf.norm(z, ord='euclidean', axis=-1, keepdims=True)
        m_l2_norm = tf.norm(m, ord='euclidean', axis=-1, keepdims=True)

        dot_prod = tf.tensordot(z, m, axes=([4], [1]))

        m_l2_norm_reshaped = tf.reshape(m_l2_norm, (1, 1, 1, 1, -1))
        norm_prod = tf.matmul(z_l2_norm,m_l2_norm_reshaped)

        #? COsine similarity 
        cosine_sim  = dot_prod / (norm_prod+self.eps)

        
        return cosine_sim

#* Layer 02 : Softmax on last axis of the similarity 
class Softmax(Layer):
# To istanciate the layer and set the kernel parameters  
    def __init__(self,type="cosine"):
        '''Initializes the instance attributes'''
        super(Softmax, self).__init__()

    # Perform the computation 
    def call(self, inputs):
        """
            Input : 
                sim: of shape (m,T,H,W,N)
            Output 
                softmax : of shape (m,T,H,W,N)
        """
        return tf.nn.softmax(inputs,axis=-1)

#* LAyer 04 : HArd Shrinkage 
class HardShrinkage(Layer):
    # To istanciate the layer and set the kernel parameters  
    def __init__(self,delta=0.0025,epsilon=1e-14):
        '''Initializes the instance attributes'''
        super(HardShrinkage, self).__init__()
        self.delta = delta
        self.epsilon = epsilon 

    # Perform the computation 
    def call(self, inputs):
        """
            Input : 
                sim: of shape (m,2,H,W,N)
            Output 
                softmax : of shape (m,2,H,W,N)
        """
        tmp = inputs-self.delta
        mask = tf.nn.relu(tmp)/ (tf.abs(tmp)+self.epsilon)
        return  mask*inputs


#* Layer 05 : Normalization of w usingord-norm 
class Normalize(Layer):
    # To istanciate the layer and set the kernel parameters  
    def __init__(self,ord=1,eps=1e-14):
        '''Initializes the instance attributes'''
        super(Normalize, self).__init__()
        self.ord = ord
        self.eps = eps

    # Perform the computation 
    def call(self, inputs):
        """
            Input : 
                z: of shape (m,2,H,W,N)
            Output 
                z_norm : of shape (m,2,H,W,N) 
        """
        norm  =   tf.norm(inputs,ord=self.ord,axis=-1,keepdims=True)+self.eps

        return   inputs / norm

#* LAyer06 : Memory output layer
class MemoryLayer(Layer):
    # To istanciate the layer and set the kernel parameters  
    def __init__(self):
        '''Initializes the instance attributes'''
        super(MemoryLayer, self).__init__()

    # Perform the computation 
    def call(self, inputs):
        """
            Input : 
                w: of shape (m,2,H,W,N)
                m : of shape (N,C)
            Output 
                mw : of shape (m,2,H,W,C) 
        """
        (w,m) = inputs 
        return    tf.tensordot(w, m, axes=([4], [0]))

#* Memory model
class Memory(tf.keras.Model):

    def __init__(self,nb_items=2000,size_item=256,delta=0.00025,epsilon=1e-14):
        """
            nb_items : number of items in the memory 
            size_item  : size of one item 
        """
        super(Memory, self).__init__(name='Memory')
        
        self.nb_items = nb_items
        self.size_item = size_item
        self.delta=delta
        self.epsilon = epsilon

        #* Items in the memory 
        stdv = 1. / math.sqrt(size_item)
        m_init = tf.random_uniform_initializer(-stdv,stdv)



        self.m = tf.Variable(name="Memory-Items",
            initial_value=m_init(shape=(nb_items,size_item),dtype='float32'),
            trainable=True)

        #* Layers of the memory 
        # LAyer 01 : Similarity 
        self.similarity_layer = Similarity()
        # LAyer 02 : Softmax 
        self.softmax_layer = Softmax() 
        # LAyer 03 : HArd Shrinkage 
        self.hard_shrinkage_layer = HardShrinkage(delta=self.delta,epsilon=self.epsilon)
        # LAyer 04 : L1-norm 
        self.norm_layer = Normalize(ord=1)
        # LAyer 05 : Mem-LAyer
        self.mem_layer = MemoryLayer()

    def call(self, input_tensor):
        
        sem = self.similarity_layer((input_tensor,self.m))
        wi = self.softmax_layer(sem)
        wi_hat = self.hard_shrinkage_layer(wi)
        wi_hat = self.norm_layer(wi_hat)
        z_hat = self.mem_layer((wi_hat,self.m))
        return z_hat,wi_hat


    def set_memory_items(self,values):
        self.m.assign(values)

    def get_memory_items(self,values):
        return tf.constant(self.m.numpy())


#* MemAE Model 
class MemAE(tf.keras.Model):

    def __init__(self,input_shape_dim=(16,16,128,128,1),nb_items=2000,size_item=256,delta=0.0025,epsilon=1e-13):
        """
            Paramters :
                input_shape_dim : shape of input tensors (None,T,H,W,1)
                nb_items : size of the memory 
                size_item : size of one item in the memory (N) 
                delta : Shrink Threshold  (recomanded value [1/N,3/N])
                epsilon : epsilon to add when applying the shrinkage (should be small)
        """


        super(MemAE, self).__init__(name='MemAE')
        
        self.input_shape_dim = input_shape_dim

        self.encoder = Encoder()
        self.memory = Memory(nb_items,size_item,delta,epsilon)
        self.decoder = Decoder()
        
        print(self.input_shape_dim)
        print((self.input_shape_dim[0],int(self.input_shape_dim[1]/8),int(self.input_shape_dim[2]/16),int(self.input_shape_dim[3]/16),self.input_shape_dim[4]))
        
        self.encoder.build(self.input_shape_dim)
        self.memory.build((self.input_shape_dim[0],int(self.input_shape_dim[1]/8),int(self.input_shape_dim[2]/16),int(self.input_shape_dim[3]/16),256))
        self.decoder.build((self.input_shape_dim[0],int(self.input_shape_dim[1]/8),int(self.input_shape_dim[2]/16),int(self.input_shape_dim[3]/16),256))
        
    

    def call(self, input_tensor):
        z = self.encoder(input_tensor)
        z_hat,w_hat = self.memory(z)
        x_hat = self.decoder(z_hat)
        # print(f"x_hat : {tf.reduce_max(x_hat,axis=[-1,-2,-3,-4,-5])}")
        # print(f"z_hat : {tf.reduce_max(z_hat,axis=[-1,-2,-3,-4,-5])}")
        # print(f"w_hat : {tf.reduce_max(w_hat,axis=[-1,-2,-3,-4,-5])}")

        return x_hat,w_hat

#TODO : light weighted version 



if __name__=="__main__":
    
    input_shape = (2,16,128,128,1)
    
    encoder = Encoder()
    encoder.build((2,16,256,256,1))
    encoder.summary()


    decoder = Decoder()
    decoder.build((None, 2, 16, 16, 256))
    decoder.summary()


    # memory = Memory()
    # memory.build((None, 2, 16, 16, 256))
    # memory.summary()


    # ae = AE((None,16,256,256,1))
    # ae.build((None,16,256,256,1))
    # ae.summary()

    # memae = MemAE((2,16,128,128,1))
    # memae.build((2,16,128,128,1))
    # memae.summary()


    # var1 = np.ones(input_shape)*4.0
    # print(memae(var1)[1])



    # # print(sim_layer(()))

    # #? Testing memory layers 
    # input_shape = (10,2,10,16,256)
    # # my_dense = Similarity(units=1)

    # var1 = tf.constant(np.ones(input_shape)*4.0)
    # l2_norm1 = tf.norm(var1, ord='euclidean', axis=-1, keepdims=True)
    # # print("l2_norm1")
    # # print(l2_norm1.shape) # 32
    # # print(l2_norm1[0,0,0,0,0]) # 32

    # #? Memory 
    # var2 = tf.constant(np.ones((2000,256))*2.0)
    # l2_norm2 = tf.norm(var2, ord='euclidean', axis=-1, keepdims=True)
    # # print("l2_norm2")
    # # print(l2_norm2.shape) #16
    # # print(l2_norm2[0,0,0,0,0]) # 32

    # #? Dot Product 
    # dot_prod = tf.tensordot(var1, var2, axes=([4], [1]))
    # # print("dot_prod")
    # # print(dot_prod.shape) #16
    # # print(dot_prod[0,0,0,0,0]) # 32

    # #? Norm product , shape(16,2,16,16,2000) and value of 512  
    # tmp1 = tf.reshape(l2_norm2, (1, 1, 1, 1, -1))
    # norm_prod = tf.matmul(l2_norm1,tmp1)
    # # print(norm_prod.shape)

    # #? COsine similarity 
    # cosine_sim  = dot_prod / norm_prod
    # # print(cosine_sim.shape)


    # sim_layer = Similarity()
    # sim = sim_layer((var1,var2))
    # print("sim")
    # print(sim.shape)
    # print(sim[0,0,0,0,0])

    # softmax_layer = Softmax()
    # softmax = softmax_layer(sim)
    # print("softmax")
    # print(softmax.shape)
    # print(softmax[0,0,0,0,0])

    # hard_shrinkage_layer = HardShrinkage(delta=0.0,epsilon=0.0)
    # hard = hard_shrinkage_layer(softmax)
    # print("hard_shrinkage_layer")
    # print(hard.shape) # 0.0005
    # print(hard[0,0,0,0,0])

    # norm_layer = Normalize(ord=1)
    # norm = norm_layer(hard)
    # print("norm_layer")
    # print(norm.shape) # 0.0005
    # print(norm[0,0,0,0,0])

    # print("prod")

    # mem_layer = MemoryLayer()
    # # prod = mem_layer((norm,var2))

    # prod = tf.tensordot(norm, tf.cast(var2, norm.dtype), axes=([4], [0]))
    
    # print(prod.shape)
    # print(prod[0,0,0,0,0])



    # mem = Memory()
    # mem.build((10,2,10,16,256))
    # mem.set_memory_items(var2.numpy())
    # print(mem(var1)[0].shape)
    # print(mem(var1)[0][0,0,0,0,0])
    # print(mem(var1)[1].shape)
    # print(mem(var1)[1][0,0,0,0,0])
    





    # norm= tf.norm(hard,ord=2,axis=-1,keepdims=True)
    # print(norm.shape)
    # print(norm[0,0,0,0,0])

    # norm = hard / norm
    # print(norm.shape)
    # print(norm[0,0,0,0,0])


    # print(np.linalg.norm(hard, ord=1,axis=-1,keepdims=True))

    # #? Relu on wi - delta  
    
    # delta=0
    # epsilon=0.1

    # relu = tf.nn.relu(softmax-delta)

    # mask = relu/ (tf.abs(softmax-delta)+epsilon)
    # print(mask)
    # # print( mask*softmax)



    # encoder = Encoder()
    # encoder.build((None,16,256,256,1))
    # encoder.summary()


    # decoder = Decoder()
    # decoder.build((None, 2, 16, 16, 256))
    # decoder.summary()

    # ae = AE((None,16,256,256,1))
    # ae.build((None,16,256,256,1))
    # ae.summary()



    # input =tf.keras.layers.Input(shape=(16,256,256,1))

    # encoder = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(16,256,256,1)),
    #     tf.keras.layers.ZeroPadding3D(padding=1),
    #     tf.keras.layers.Conv3D(96,(3, 3, 3),(1, 2, 2),data_format="channels_last"),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.LeakyReLU(0),
    #     tf.keras.layers.ZeroPadding3D(padding=1),
    #     tf.keras.layers.Conv3D(128,(3, 3, 3),(2,2,2),data_format="channels_last"),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.LeakyReLU(0),
    #     tf.keras.layers.ZeroPadding3D(padding=1),
    #     tf.keras.layers.Conv3D(256,(3, 3, 3),(2,2,2),data_format="channels_last"),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.LeakyReLU(0),
    #     tf.keras.layers.ZeroPadding3D(padding=1),
    #     tf.keras.layers.Conv3D(256,(3, 3, 3),(2, 2, 2),data_format="channels_last"),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.LeakyReLU(0),
    # ])
    # encoder.summary()


    # decoder = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(2,16,16,256)),
    #     tf.keras.layers.Conv3DTranspose(256,kernel_size	= (3, 3, 3),strides=(2, 2, 2),data_format="channels_last",padding='same'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.LeakyReLU(0),
    #     tf.keras.layers.Conv3DTranspose(128,kernel_size	= (3, 3, 3),strides=(2, 2, 2),data_format="channels_last",padding='same'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.LeakyReLU(0),
    #     tf.keras.layers.Conv3DTranspose(96,kernel_size	= (3, 3, 3),strides=(2, 2, 2),data_format="channels_last",padding='same'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.LeakyReLU(0),
    #     tf.keras.layers.Conv3DTranspose(1,kernel_size	= (3, 3, 3),strides=(1, 2, 2),data_format="channels_last",padding='same'),

    # ])
    # # decoder.summary()

    # # Defining model 
    # z = encoder(input)
    # x_hat = decoder(z)
    # model  = tf.keras.Model(inputs=input,outputs=x_hat)

    # model.summary()
