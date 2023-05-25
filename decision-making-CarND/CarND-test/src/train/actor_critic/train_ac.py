import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from environment import Environment
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config))

def main():
    env = Environment()
    env.run()
    

if __name__ == '__main__':
    main()