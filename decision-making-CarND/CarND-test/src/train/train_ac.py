import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from environment import Environment

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config))

'''
    ## Figure out if you need to fit a model for actor critic
        self.mode.fit(state, target, epochs=1, verbose=0)
    
    ## Decay the rate for every iteration of an episode
        if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
'''

def main():
    env = Environment()
    env.run()
    

if __name__ == '__main__':
    main()