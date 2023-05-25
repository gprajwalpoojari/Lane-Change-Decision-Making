from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class Critic:
    def __init__(self, state_height, state_width, value_size, learning_rate):
        self.state_height = state_height
        self.state_width = state_width
        self.value_size = value_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.z = [np.zeros(arr.shape) for arr in self.model.trainable_variables]

    # Neural Net for Critic Model. Takes the current state and outputs future value of the state
    def _build_model(self):
        input1 = Input(shape=(1, self.state_height, self.state_width))
        conv1 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
                       input_shape=(1, self.state_height, self.state_width))(input1)
        conv2 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid')(conv1)
        conv3 = Conv2D(3, 1, strides=1, activation='relu', padding='valid')(conv2)
        state1 = Flatten()(conv3)
        input2 = Input(shape=(3,))
        state2 = concatenate([input2, state1])
        state2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.9))(state2)
        state2 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.9))(state2)
        out_put = Dense(self.value_size)(state2)
        model = Model(inputs=[input1, input2], outputs=out_put)
        self.optimizer = Adam(lr=self.learning_rate)
        return model

    def train(self, gamma, lamda, next_state, curr_state, immediate_reward):
        with tf.GradientTape() as critictape:
            curr_state_value = self.model(curr_state)
            next_state_value = self.model(next_state)
            advantage = immediate_reward + gamma * next_state_value - curr_state_value
            loss = advantage ** 2
        gradient = critictape.gradient(loss, self.model.trainable_variables)
        # print("#################################CRITIC VALUE#################################")
        # print(value)
        # print(gradient[0])
        updated_gradient = []
        for i in range(len(self.z)):
            self.z[i] = gamma * lamda * self.z[i] + gradient[i]
            # if (i == 0):
                # print("#################################CRITIC GRADIENT AFTER DISCOUNT#################################")
                # print(self.z[i])
            updated_gradient.append(self.z[i] * advantage[0][0])
            # if (i == 0):
                # print("#################################UPDATED GRADIENT#################################")
                # print(updated_gradient[i])
        self.optimizer.apply_gradients(zip(updated_gradient, self.model.trainable_variables))
        # self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)