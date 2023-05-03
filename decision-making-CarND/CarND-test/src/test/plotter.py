from matplotlib import pyplot as plt
import csv



num_lane_changes_list = []
num_collisions_list = []
avg_speed_list = []
file_name = open('/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/test/test_metric.csv', 'r')
data = csv.DictReader(file_name)
for col in data:
    num_lane_changes_list.append(col['num_lane_changes'])
    num_collisions_list.append(col['num_collisions'])
    avg_speed_list.append(col['avg_speed'])

num_lane_changes_list = [float(i) for i in num_lane_changes_list]
speed_list = [float(i) for i in avg_speed_list]
num_collisions_list = [float(i) for i in num_collisions_list]

X = [i for i in range(1, len(num_collisions_list)+1)]

plt.plot(X, num_collisions_list, label='Collisions per episode')
plt.xlabel('Numer of Episode')
plt.ylabel('Number of collisions')
plt.legend()
plt.show()

plt.plot(X, num_lane_changes_list, label='Lane change per episode')
avg_lane_change = sum(num_lane_changes_list) / len(num_lane_changes_list)
avg_lane_change_list = [avg_lane_change] * len(num_lane_changes_list)
plt.plot(X, avg_lane_change_list, label='Average lane changes for 20 episodes')
plt.xlabel('Numer of Episode')
plt.ylabel('Number of lane changes')
plt.legend()
plt.show()

plt.plot(X, speed_list, label='Average speed per episode')
avg_speed = sum(speed_list) / len(avg_speed_list)
avg_speed_list = [avg_speed] * len(speed_list)
plt.plot(X, avg_speed_list, label='Average speed for 20 episodes')
plt.xlabel('Numer of Episode')
plt.ylabel('Number of Average Collisions')
plt.legend()
plt.show()

