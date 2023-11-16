import math
import time

import numpy as np
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
from numpy.linalg import norm


# size=10

class Environment:
    def __init__(self, size, targets, alpha):
        self.size = size
        self.targets_locations = []
        self.alpha = alpha
        self.signals = np.zeros((self.size, self.size))
        self.p_s = np.full((self.size, self.size), 1.0)
        self.memo = {}
        self.agents = []
        self.IG_KL = np.zeros((self.size, self.size))
        self.IG_Normalization = np.zeros((self.size, self.size))
        self.PTA = 1
        for target in range(targets):
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            if (x, y) not in self.targets_locations:
                self.targets_locations.append((x, y))
            else:
                x = np.random.randint(self.size)
                y = np.random.randint(self.size)
                if (x, y) not in self.targets_locations:
                    self.targets_locations.append((x, y))
                else:
                    x = np.random.randint(self.size)
                    y = np.random.randint(self.size)
                    self.targets_locations.append((x, y))

    def generate_signals(self):
        signals = np.random.rand(self.size, self.size)
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.targets_locations:
                    self.signals[i][j] = 1
                    continue
                else:
                    if signals[i][j] < self.alpha:
                        self.signals[i][j] = 1
                    else:
                        self.signals[i][j] = 0
        return self.signals

    def merge_info(self, t):
        for i in range(self.size):
            for j in range(self.size):
                mul = 1
                divide1 = 1
                divide2 = 1
                for agent in self.agents:
                    mul *= agent.p_s[i][j]
                    divide1 *= agent.p_s[i][j]
                    divide2 *= (1 - agent.p_s[i][j])
                result = mul / (divide1 + divide2)
                self.p_s[i][j] = result
                self.memo[(t, i, j)] = result
        return self.p_s

    def calc_IG_KL(self, t):
        self.IG_KL = np.zeros((self.size, self.size))
        agents_locations = []

        for agent in self.agents:
            agents_locations.append(agent.location)
        for i in range(self.size):
            for j in range(self.size):
                try:
                    if t == 0:
                        return 0
                    if (i, j) in agents_locations:
                        self.IG_KL[i][j] += 0
                        continue
                    else:
                        a = self.p_s[i][j] * math.log2(self.p_s[i][j] / self.memo[(t - 1, i, j)])
                        b = (1 - self.p_s[i][j]) * math.log2((1 - self.p_s[i][j]) / (1 - self.memo[(t - 1, i, j)]))
                        self.IG_KL[i][j] += a + b
                except ValueError:
                    print("Value error:" + str((a, b, self.p_s[i][j], self.memo[(t - 1, i, j)])))
                    self.IG_KL[i][j] += 0

                    continue
        self.IG_KL[np.isnan(self.IG_KL)] = 0
        return self.IG_KL.sum()

    def calc_IG_Normalizatrion(self, t):
        self.IG_Normalization = np.zeros((self.size, self.size))
        agents_locations = []
        for agent in self.agents:
            agents_locations.append(agent.location)
        normalized_array = self.p_s / self.p_s.sum()
        normalized_memo_sum = 0
        if t != 0:
            for i in range(self.size):
                for j in range(self.size):
                    normalized_memo_sum += self.memo[(t - 1, i, j)]

        for i in range(self.size):
            for j in range(self.size):

                try:
                    if t == 0:
                        return 0
                    if (i, j) in agents_locations:
                        self.IG_Normalization[i][j] += 0
                        continue
                    else:
                        normalized_memo = (self.memo[(t - 1, i, j)]) / normalized_memo_sum
                        self.IG_Normalization[i][j] += normalized_array[i, j] * math.log2(
                            (normalized_array[i][j]) / (normalized_memo))
                except ValueError:
                    print("Value error:" + str((normalized_memo, self.IG_Normalization[i][j], self.p_s[i][j], self.memo[(t - 1, i, j)])))
                    continue
                    self.IG_Normalization[i][j] += 0
        self.IG_Normalization[np.isnan(self.IG_Normalization)] = 0
        return self.IG_Normalization.sum()

    def calc_cog(self, method):
        d = {}
        cog_x = 0
        cog_y = 0
        if method == 'KL':
            for i in range(self.size):
                for j in range(self.size):
                    cog_x += self.IG_KL[i][j] * i
                    cog_y += self.IG_KL[i][j] * j
            cog_x = cog_x / self.IG_KL.sum()
            cog_y = cog_y / self.IG_KL.sum()

        else:
            for i in range(self.environment.size):
                for j in range(self.environment.size):
                    cog_x += (self.IG_Normalization[i][j] * i)
                    cog_y += (self.IG_Normalization[i][j] * j)
            cog_x = cog_x / self.IG_Normalization.sum()
            cog_y = cog_y / self.IG_Normalization.sum()
        return round(cog_x), round(cog_y)


class Agent:
    def __init__(self, location, environment):
        self.location = location
        self.environment = environment
        self.sensors = []
        self.r = np.zeros((self.environment.size, self.environment.size))
        for i in range(self.environment.size):
            for j in range(self.environment.size):
                self.r[i][j] = np.sqrt((i - self.location[0]) ** 2 + (j - self.location[1]) ** 2)
        # self.received_signals=np.zeros((self.environment.size, self.environment.size))
        self.p_s = np.full((self.environment.size, self.environment.size), 1.0)
        self.next_p_s = np.zeros((self.environment.size, self.environment.size))

    def __repr__(self):
        return ('agent location: ' + str(self.location))

    def calc_distances(self):
        for i in range(self.environment.size):
            for j in range(self.environment.size):
                self.r[i][j] = np.sqrt((i - self.location[0]) ** 2 + (j - self.location[1]) ** 2)

    def forward_cog(self, cog_x, cog_y):
        # calculate the distances to the cell#
        cog = [cog_x, cog_y]
        if (cog_x, cog_y) == self.location:
            print("stayed in the same location")
            return self.location
        index_array = np.indices((self.environment.size, self.environment.size))
        reshaped_coordinate_array = np.transpose(index_array, [1, 2, 0])
        distances = norm(reshaped_coordinate_array - cog, axis=2)
        options = []
        for i in range(self.location[0] - 1, self.location[0] + 2):
            for j in range(self.location[1] - 1, self.location[1] + 2):
                if 0 <= i < self.environment.size and 0 <= j < self.environment.size and (i != cog_x or j != cog_y):
                    options.append(([i, j], distances[i, j]))
        min_distance = np.argmin([x[1] for x in options])
        next = options[min_distance][0]
        self.location = next
        print("new_location:" + str(self.location))
        self.calc_distances()
        return self.location
    def forward_eig(self):
        next = self.calc_eig()
        # if next == self.location:
        #     print("stayed in the same location")
        #     return self.location
        # else:
        self.location = next
        return self.location
    def calc_eig(self, method='KL'):
        options = []
        for i in range(self.location[0]-1, self.location[0] + 2):
            for j in range(self.location[1]-1, self.location[1] + 2):
                if 0 <= i < self.environment.size and 0 <= j < self.environment.size and (i !=  self.location[0]  or j !=  self.location[1] ):
                    options.append([i, j])
        for option in options:
            self.location = option
            self.r = np.zeros((self.environment.size, self.environment.size))
            for i in range(self.environment.size):
                for j in range(self.environment.size):
                    self.r[i][j] = np.sqrt((i - self.location[0]) ** 2 + (j - self.location[1]) ** 2)
            mul = 1
            divide1 = 1
            divide2 = 1
            for sensor in self.sensors:
                sensor.calc_eig_probability(option)
                mul *= sensor.next_p_s[option[0]][option[1]]
                divide1 *= sensor.next_p_s[option[0]][option[1]]
                divide2 *= (1 - sensor.next_p_s[option[0]][option[1]])
            result = mul / (divide1 + divide2)
            # print("result:" + str(result))
            # print("------------")
            self.next_p_s[option[0]][option[1]] = result

        best_option = list(np.unravel_index(np.argmax(self.next_p_s), self.next_p_s.shape))
        self.location = best_option
        self.r = np.zeros((self.environment.size, self.environment.size))
        for i in range(self.environment.size):
            for j in range(self.environment.size):
                self.r[i][j] = np.sqrt((i - self.location[0]) ** 2 + (j - self.location[1]) ** 2)
        self.next_p_s = np.zeros((self.environment.size, self.environment.size))
        print ("moving to ", best_option)
        return best_option

    def merge_info(self):
        for i in range(len(self.environment.signals)):
            for j in range(len(self.environment.signals)):
                mul = 1
                divide1 = 1
                divide2 = 1
                for sensor in self.sensors:
                    mul *= sensor.p_s[i][j]
                    divide1 *= sensor.p_s[i][j]
                    divide2 *= (1 - sensor.p_s[i][j])

                result = mul / (divide1 + divide2)
                # print("result:" + str(result))
                # print("------------")
                self.p_s[i][j] = result
        return self.p_s


class Sensor:
    def __init__(self, k_power, agent):
        self.k_power = k_power
        self.agent = agent
        self.r = np.zeros((self.agent.environment.size, self.agent.environment.size))
        for i in range(self.agent.environment.size):
            for j in range(self.agent.environment.size):
                self.r[i][j] = np.sqrt((i - self.agent.location[0]) ** 2 + (j - self.agent.location[1]) ** 2)
        self.received_signals = np.zeros((self.agent.environment.size, self.agent.environment.size))
        self.p_s = np.full((self.agent.environment.size, self.agent.environment.size), 0.1)
        self.IG_KL = np.zeros((self.agent.environment.size, self.agent.environment.size))
        self.next_p_s =np.zeros((self.agent.environment.size,self.agent.environment.size))




    def interpret_signals_received(self):
        # random whether the signal has been received or not
        p = np.random.rand(self.agent.environment.size, self.agent.environment.size)
        # print ("P for signals to be received is:")
        # print (p)
        p_received_signals = np.zeros((self.agent.environment.size, self.agent.environment.size))

        for i in range(self.agent.environment.size):
            for j in range(self.agent.environment.size):
                if self.agent.environment.signals[i][j] == 1:
                    p_received_signals[i][j] = np.exp(-(self.r[i][j] / self.k_power))
                else:
                    p_received_signals[i][j] = 0
        self.received_signals = np.where(p < p_received_signals, 1, 0)
        return self.received_signals
    def calc_IG_KL(self, t):
        self.IG_KL = np.zeros((self.size, self.size))
        agents_locations = []

        for agent in self.agents:
            agents_locations.append(agent.location)
        for i in range(self.size):
            for j in range(self.size):
                try:
                    if t == 0:
                        return 0
                    if (i, j) in agents_locations:
                        self.IG_KL[i][j] += 0
                        continue
                    else:
                        a = self.p_s[i][j] * math.log2(self.p_s[i][j] / self.memo[(t - 1, i, j)])
                        b = (1 - self.p_s[i][j]) * math.log2((1 - self.p_s[i][j]) / (1 - self.memo[(t - 1, i, j)]))
                        self.IG_KL[i][j] += a + b
                except ValueError:
                    print("Value error:" + str((a, b, self.p_s[i][j], self.memo[(t - 1, i, j)])))
                    self.IG_KL[i][j] += 0

                    continue
        self.IG_KL[np.isnan(self.IG_KL)] = 0
        return self.IG_KL

    def interpret_targets_bayes(self, specific= None, positive = False):
        self.received_signals = self.interpret_signals_received()
        PTA = 1
        # first time calculation

        if specific != None:
            if positive == True:
                P_target_exists_signal_received = (self.p_s[specific[0]][specific[1]]) / (self.p_s[specific[0]][specific[1]] + \
                                                                  (1 - self.p_s[specific[0]][specific[1]]) * self.agent.environment.alpha)
                probability = P_target_exists_signal_received
            else:
                P_target_exists_signal_not_received = (
                        self.p_s[specific[0]][specific[1]] * ((1 - PTA) + PTA * (1 - np.exp(-(self.r[specific[0]][specific[1]] / self.k_power)))) / \
                        (self.p_s[specific[0]][specific[1]] * ((1 - PTA) + PTA * (1 - np.exp(-(self.r[specific[0]][specific[1]] / self.k_power)))) + \
                         (1 - self.p_s[specific[0]][specific[1]]) * ((
                                                         1 - self.agent.environment.alpha * PTA) + self.agent.environment.alpha * PTA * (
                                                         1 - np.exp(-(self.r[specific[0]][specific[1]] / self.k_power))))))
                probability = P_target_exists_signal_not_received
            return probability
        else:
            for i in range(len(self.agent.environment.signals)):
                for j in range(len(self.agent.environment.signals)):
                    if self.received_signals[i][j] == 1:
                        P_target_exists_signal_received = (self.p_s[i][j]) / (self.p_s[i][j] + \
                                                                              (1 - self.p_s[i][
                                                                                  j]) * self.agent.environment.alpha)
                        self.p_s[i][j] = P_target_exists_signal_received
                    else:
                        P_target_exists_signal_not_received = (
                                self.p_s[i][j] * ((1 - PTA) + PTA * (1 - np.exp(-(self.r[i][j] / self.k_power)))) / \
                                (self.p_s[i][j] * ((1 - PTA) + PTA * (1 - np.exp(-(self.r[i][j] / self.k_power)))) + \
                                 (1 - self.p_s[i][j]) * ((
                                                                 1 - self.agent.environment.alpha * PTA) + self.agent.environment.alpha * PTA * (
                                                                 1 - np.exp(-(self.r[i][j] / self.k_power))))))
                        self.p_s[i][j] = P_target_exists_signal_not_received

            return self.p_s
    def calc_eig_probability(self,option):
        EIG_lst = []
        p_positive_signal = self.agent.environment.memo[(t, option[0], option[1])] * self.agent.environment.PTA * np.exp(
            -self.r[option[0]][option[1]] / self.k_power) + (1 - self.agent.environment.memo[
            (t, option[0], option[1])]) * self.agent.environment.alpha * self.agent.environment.PTA * np.exp(-self.r[option[0]][option[1]] / self.k_power)
        p_negative_signal = 1 - p_positive_signal

        self.next_p_s[option[0]][option[1]] = p_negative_signal * self.KL(option) + p_negative_signal * self.KL(option, positive=False)
        result = self.next_p_s[option[0]][option[1]]
        return result
    def KL(self, option, positive = True):
        a = self.interpret_targets_bayes(option,positive = positive)*math.log2(self.interpret_targets_bayes(option,positive = positive) / self.p_s[option[0]][option[1]])
        b = (1 - self.interpret_targets_bayes(option,positive = positive)) * math.log2((1 -self.interpret_targets_bayes(option,positive = positive)) / (1 -  self.p_s[option[0]][option[1]]))
        result = a + b
        return result





agents = []
env = Environment(size=20, targets=3, alpha=0.3)
env.targets_locations = [(2, 2), (15, 5), (10, 16)]
agent1 = Agent((9, 10), environment=env)
agents.append(agent1)
env.agents = agents
# agent2=Agent((np.random.randint(env.size),np.random.randint(env.size)),environment=env)
sensor1 = Sensor(k_power=10, agent=agent1)
# sensor2=Sensor(k_power=10,agent=agent1)
# sensor3=Sensor(k_power=10,agent=agent2)
# sensor4=Sensor(k_power=10,agent=agent2)
# agent2.sensors=[sensor3,sensor4]
agent1.sensors = [sensor1]
env.agents = [agent1]

final_results = []
T = 100
agents = [agent1]
total_IG_KL = []
total_IG_Normalized = []
plt.title("Targets Map")
plt.ion()
plt.show()
fig, ax = plt.subplots()
# ax1 = axs[0]
# ax2 = axs[1]


for t in range(T):
    for agent in agents:
        for sensor in agent.sensors:
            x = np.zeros((env.size, env.size))
            signals = env.generate_signals()
            sensor.p_s = sensor.interpret_targets_bayes()
        agent.merge_info()
        matrix = agent.p_s
    final_matrix = env.merge_info(t)
    total_IG_KL.append(env.calc_IG_KL(t))
    total_IG_Normalized.append(env.calc_IG_Normalizatrion(t))
    if t > 0:
        cog_x, cog_y = env.calc_cog('KL')
        for agent in env.agents:
            print("location before movement:" + str(agent.location))
            # agent.forward_cog(cog_x, cog_y)

            agent.forward_eig()
            cog_step = np.zeros((env.size, env.size))
            eig_step = np.zeros((env.size, env.size))

            # cog_step[agent.location[0],agent.location[1]] = 1
            eig_step[agent.location[0], agent.location[1]] = 1
            cog_show = cog_step.copy()
            eig_show = eig_step.copy()
            # ax.spy(cog_show, markersize=6, color='green')


            ax.spy(eig_show, markersize=6, color = 'green')

            plt.imshow(final_matrix, cmap='hot', interpolation='gaussian')
            plt.tight_layout()
            plt.gcf().canvas.draw()  # update display window
            plt.pause(0.000000001)
            plt.tight_layout()
            x = np.zeros((env.size, env.size))
            ax.clear()
plt.close()


def plot_IG(input_list, criterion, time=T):
    t_array = np.arange(0, time, 1)
    s = input_list
    fig, ax = plt.subplots()
    ax.plot(t_array, s)

    ax.set(xlabel='time (s)', ylabel='IG',
           title=f'Information Gain {criterion}')
    ax.grid()

    fig.savefig(fr"{criterion}.png")

    plt.show()
    return


plot_IG(total_IG_KL, 'KL', time=T)
plot_IG(total_IG_Normalized, 'Normalized', time=T)

# print (env.p_s)
print(env.targets_locations)
print("agent location:")

print(agent1.location)

# print ("----------")
#

# print (agent1.p_s)
# print ("--------------")
# print ("received signals:")
# print (sensor1.received_signals)
# print ("r:")
# print(agent1.r)
### continue: insert the KL formula into the calculation!###