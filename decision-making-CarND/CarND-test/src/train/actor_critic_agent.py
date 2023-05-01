from actor import Actor
from critic import Critic

class ActorCriticAgent:
    def __init__(self, state_height, state_width):
        self.state_height = state_height
        self.state_width = state_width
        self.gamma = 0.90  # discount rate
        self.lamda = 0.90  # eligibility trace decay
        self.epsilon = 0.3  # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9  # init with pure exploration
        self.learning_rate = 0.00025
        self.i = 1
        self.action_size = 3
        self.val_size = 1
        self.actor = Actor(state_height, state_width, self.action_size, self.learning_rate)
        self.critic = Critic(state_height, state_width, self.val_size, self.learning_rate)
        self.path = "/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/episode/"

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
    
    def update(self, prev_state, curr_state, immediate_reward):
        advantage = immediate_reward + self.gamma * self.critic.model.predict(
                        curr_state) - self.critic.model.predict(prev_state)
        self.actor.train(self.gamma, self.lamda, advantage)
        self.critic.train(self.gamma, self.lamda, advantage)
        action = self.actor.act(curr_state)
        self.i = self.gamma * self.i
        return action
