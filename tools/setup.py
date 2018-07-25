import tensorflow as tf

from keras import backend as K


def setup_tensorflow():
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)
    K.set_session(sess)
