from actor import Actor
from critic import Critic
import numpy as np

class ActorCriticAgent:
    def __init__(self, state_height, state_width):
        self.state_height = state_height
        self.state_width = state_width
        self.gamma = 0.90  # discount rate
        self.lamda = 0.90  # eligibility trace decay
        self.learning_rate = 0.00025
        self.action_size = 3
        self.val_size = 1
        self.actor = Actor(state_height, state_width, self.action_size, self.learning_rate)
        self.critic = Critic(state_height, state_width, self.val_size, self.learning_rate)
        self.path = "/home/prajwal/Lane-Change-Decision-Making/decision-making-CarND/CarND-test/src/train/episode/"

    def load(self, episode):
        actor_path = self.path + '/actor/' + str(episode) + ".h5"
        critic_path = self.path + '/critic/' + str(episode) + ".h5"
        self.actor.model.load_weights(actor_path)
        self.critic.model.load_weights(critic_path)

    def save(self, episode):
        actor_path = self.path + '/actor/' + str(episode) + ".h5"
        critic_path = self.path + '/critic/' + str(episode) + ".h5"
        self.actor.model.save_weights(actor_path)
        self.critic.model.save_weights(critic_path)
    
    def update(self, curr_state, next_state, immediate_reward):
        next_state_value = self.critic.model.predict(next_state)
        curr_state_value = self.critic.model.predict(curr_state)
        advantage = immediate_reward + self.gamma * next_state_value - curr_state_value
        print("#################################REWARD NEXT_VAL CURR_VAL ADVANTAGE#################################")
        print(immediate_reward, next_state_value, curr_state_value, advantage)
        self.actor.train(self.gamma, self.lamda, advantage)
        self.critic.train(self.gamma, self.lamda, advantage, curr_state)
    
    def reinitialize(self):
        self.actor.z = [np.zeros(arr.shape) for arr in self.actor.model.trainable_variables]
        self.critic.z = [np.zeros(arr.shape) for arr in self.critic.model.trainable_variables]
        self.actor.i = 1

        