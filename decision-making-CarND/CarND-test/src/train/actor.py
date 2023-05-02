from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np



class Actor:
    def __init__(self, state_height, state_width, action_size, learning_rate):
        self.state_height = state_height
        self.state_width = state_width
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.i = 1
        self.z = [np.zeros(arr.shape) for arr in self.model.trainable_variables]

    # Neural Net for Actor Model. Takes the current state and outputs probabilities of actions
    def _build_model(self):
        input1 = Input(shape=(1, self.state_height, self.state_width))
        conv1 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
                       input_shape=(1, self.state_height, self.state_width))(input1)
        conv2 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid')(conv1)
        conv3 = Conv2D(3, 1, strides=1, activation='relu', padding='valid')(conv2)
        state1 = Flatten()(conv3)
        input2 = Input(shape=(3,))
        state2 = concatenate([input2, state1])
        state2 = Dense(256, activation='relu')(state2)
        state2 = Dense(64, activation='relu')(state2)
        out_put = Dense(self.action_size, activation='softmax')(state2)
        model = Model(inputs=[input1, input2], outputs=out_put)
        self.optimizer = Adam(lr=self.learning_rate)
        return model


    def train(self, gamma, lamda, advantage):
        # print("#################################ACTOR LOSS#################################")
        # print(self.loss)
        # print(gradient)
        updated_gradient = []
        for i in range(len(self.z)):
            self.z[i] = gamma * lamda * self.z[i] + self.i * self.gradient[i]
            updated_gradient.append(self.z[i] * advantage[0][0])
        self.i = gamma * self.i
        self.optimizer.apply_gradients(zip(updated_gradient, self.model.trainable_variables))

    # Provides an action given a current state
    def act(self, state):
        with tf.GradientTape() as actortape:
            action_distribution = self.model(state)
            act_value = np.random.choice(self.action_size, p=np.squeeze(action_distribution))
            loss = tf.math.log(action_distribution[0, act_value])
        self.gradient = actortape.gradient(loss, self.model.trainable_variables)
        return act_value  # returns action

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
