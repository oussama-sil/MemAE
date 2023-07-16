import tensorflow as tf




#* WiderResNet-38 





if __name__ == '__main__':
    resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet' )

    resnet.summary()

