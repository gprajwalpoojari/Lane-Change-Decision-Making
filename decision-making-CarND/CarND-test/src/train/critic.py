from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam



class Critic:
    def __init__(self, state_height, state_width, value_size, learning_rate):
        self.state_height = state_height
        self.state_width = state_width
        self.value_size = value_size
        self.learning_rate = learning_rate
        self.theta = 0
        self.model = self._build_model()

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
        state2 = Dense(256, activation='relu')(state2)
        state2 = Dense(64, activation='relu')(state2)
        out_put = Dense(self.value_size)(state2)
        model = Model(inputs=[input1, input2], outputs=out_put)
        self.optimizer = Adam(lr=self.learning_rate)
        return model
    
    def calculate_gradient(self):
        return 0

    def train(self, gamma, lamda, advantage):
        gradient = self.calculate_gradient()
        self.theta = gamma * lamda * self.theta + gradient
        self.optimizer.apply_gradients(zip(self.theta * advantage, self.model.trainable_variables))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)