# Lane-Change-Decision-Making-using-Reinforcement-Learning

## About the project
This project utilizes a three-lane highway traffic simulator for reinforcement learning based lane change decision making. The simulator provides current vechicle position and surrounding vehicles position information. Deep Q Learning Network is employed for the decision making model. The action space is discretized into 3 actions, namely, stay on the current lane, make left lane change and make right lane change. 

## Snapshot of the simulator
The simulator is taken from one of the udacity courses. Here is a snapshot of the simulator:

![Screenshot from 2023-05-25 14-37-52](https://github.com/gprajwalpoojari/Lane-Change-Decision-Making/assets/53962958/7fd34230-7893-413e-8fcd-1da966875323)

## Training phase
In the initial phases, the Model takes random actions that result in unnecessary lane changes and collision. The network is penalized for unnecessary lane changes and lane changes that result in collision. Here is an example:

![train](https://github.com/gprajwalpoojari/Lane-Change-Decision-Making/assets/53962958/99d91822-7803-490e-aaa6-d691944b8b34)


## Final model output
After training the network, the model makes lane changes only when required. Here is an example:

![test](https://github.com/gprajwalpoojari/Lane-Change-Decision-Making/assets/53962958/4e37c7cc-5b5e-4e33-bfaf-b46db4238534)

## References
https://arxiv.org/pdf/1904.00231.pdf


