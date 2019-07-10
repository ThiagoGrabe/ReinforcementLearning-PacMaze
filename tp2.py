import numpy as np
import time
import random
import os
import sys
import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Enviroment import PacMazeWorld
from Agent import Agent

GAMMA = 0.9
# ALPHA = .3
# EPSILON = 0.9
#
# EPISODES = -1

mapa = None
agent = None

path = None

# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# FOLDER2SAVE = str(DIR_PATH+'/results/'+str(EPISODES)+'ep_'+str(GAMMA)+'gamma'+str(ALPHA)+'alpha'+str(EPSILON)+'epsilon')


def initWorld(world):

    map = PacMazeWorld(world)
    rows, columns = map.worldShape()
    map.world = np.array(map.reset()).reshape(rows, columns)
    return map


def insert_agent(mapa):

    a = np.where(mapa.world == '-')
    idx = random.randrange(0, len(a[0]))
    i, j = a[0][idx], a[1][idx]

    agent = Agent(i, j) # Creating the RL agent

    agent.x, agent.y = i, j
    return agent


def getState(map, agent):
    i, j = agent.getPose()
    vec_x, vec_y = map.getPossibleStates()

    counter = 0
    for x, y in zip(vec_x, vec_y):
        if x == i and y == j:
            return counter
        else:
            counter += 1
    print('Should you be here?')


def getAction(Q, state, epsilon):
    if random.random() > epsilon:
        choice = random.randint(0, 3)
        return choice
        # return choice, Q[state, choice]
    else:
        choice = np.argmax(Q[state, :])
        return choice


def AgentMovement(map, agent, move):
    i, j = agent.getPose()
    movement = map.actions[move]
    new_i, new_j = agent.getNewPosition(movement)
    vec_x, vec_y = map.getPossibleStates()

    counter = -1
    for x, y in zip(vec_x, vec_y):
        if x == new_i and y == new_j:
            counter += 1

    if counter == -1:
        agent.x, agent.y = i, j


def getReward(map, agent):
    i, j = agent.getPose()

    map_avatar = map.world[i][j]

    if map_avatar == map.wall:
        print('Should you be here?')
    elif map_avatar == map.empty:
        reward = -1
    elif map_avatar == map.gost:
        reward = -10
    elif map_avatar == map.gold:
        reward = 10

    return reward


def IsTerminal(map, agent):
    location = agent.getPose()
    a = map.getGoldPose()
    b = map.getGostPose()

    vec_x, vec_y = map.getGostPose()
    if len(vec_x) > 1 or len(vec_y) > 1:
        counter = -1
        for x, y in zip(vec_x, vec_y):
            if x == location[0] and y == location[1]:
                counter += 1
        if counter != -1:
            return False
        else:
            pass
    else:
        if agent.getPose() == map.getGostPose():
            return False
        else:
            pass

    vec_x2, vec_y2 = map.getGoldPose()
    if len(vec_x2) > 1 or len(vec_y2) > 1:
        counter = -1
        for x, y in zip(vec_x2, vec_y2):
            if x == location[0] and y == location[1]:
                counter += 1
        if counter != -1:
            return False
        else:
            pass
    else:
        if agent.getPose() == map.getGoldPose():
            return False
        else:
            return True


def export(map, agent, table):
    with open('q.txt', "w+") as file:
        vec_x, vec_y = map.getPossibleStates()
        col = 0
        for x, y in zip(vec_x, vec_y):
            act = 0
            for val in table[col]:
                value = round(val, 3)
                # print('The value is:',value)
                line = (str(x) + ',' + str(y) + ',' + str(map.actions[act][0]) + ',' + str(value))
                act += 1
                avatar = map.getAgentState(x, y)
                if avatar == map.empty:
                    file.write(line + '\n')
                # print(line)

            col += 1
        file.close()

        with open('pi.txt', "w+") as file2:
            vec_x, vec_y = map.getPossibleStatesTOPRINT()
            E = table
            teste = 0
            for world_row, row in enumerate(map.world):

                line = ''
                for id, element in enumerate(row):
                    if element == map.wall:
                        line = line+str(map.wall)
                    elif element == map.gost:
                        line = line+str(map.gost)
                        teste += 1
                    elif element == map.gold:
                        line = line+str(map.gold)
                        teste += 1
                    elif element == map.empty:
                        choice = np.argmax(table[teste, :])
                        value = np.max(table[teste, :])
                        a = value
                        letter = map.actions[choice]
                        line = line + str(letter[0])
                        teste += 1
                file2.write(line + '\n')
                # print(line)
            file2.close()

        # np.savetxt(fname='qtable.txt',X=table, fmt='%.5f', delimiter=" ")

#
# def plots(x, y, description, color, legenda=''):
#
#     plt.plot(x, y, label=legenda, color=color)
#     plt.title(description)
#     plt.legend(loc='best')
#     # plt.xlim([0, np.max(x)])
#     # plt.ylim([0, np.max(y)])
#     plt.savefig(FOLDER2SAVE+'/'+str(description)+'.png', dpi=300)
#     plt.clf()
#
#
# def saveVec(vec, name):
#     np.savetxt(fname=FOLDER2SAVE+'/vector_'+str(name)+'.txt', X=vec, fmt='%.5f', delimiter=" ")


def run(txt, exp_rewards):
    TEXT = str(txt)
    map = initWorld(TEXT)
    # STATES = np.where(initWorld('pacmaze.txt') == '-')
    STATES = map.getPossibleStates()[0]
    ACTIONS = map.actions

    # Q = np.random.rand(len(STATES), len(ACTIONS))
    Q = np.zeros((len(STATES), len(ACTIONS)))

    j = Q

    a = map.getPossibleStates()

    # store the training progress of this algorithm for each episode
    episode_rewards = []
    episode_times = []
    episode_epsilon = []

    # solve the environment over certain amount of episodes
    for episode in range(1, EPISODES+1):

        # log the training start
        training_start = time.time()
        # reset the environment, rewards, and steps for the new episode
        map = initWorld(TEXT)

        epsilon = EPSILON

        # place the agent
        agent = insert_agent(map)

        # init reward equal zero
        episode_reward = 0
        while IsTerminal(map, agent):

            # print(count_step)

            state = getState(map, agent)

            action = getAction(Q, state, epsilon)
            AgentMovement(map, agent, action)

            reward = getReward(map, agent)
            episode_reward += reward

            next_state = getState(map, agent)

            Q_target = reward + GAMMA * np.max(Q[next_state, :])
            Q_delta = Q_target - Q[state, action]
            Q[state, action] = Q[state, action] + (ALPHA * Q_delta)

            # Q[state, action] = Q[state, action] + ALPHA * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

            if epsilon > 0.1:
                epsilon -= 1/episode

        episode_time = time.time() - training_start
        episode_list = [i for i in range(0, EPISODES)]
        episode_rewards.append(episode_reward)
        episode_times.append(episode_time)
        episode_epsilon.append(epsilon)

        # FACTOR = int(EPISODES/10)
        # np.mean(np.array(episode_rewards).reshape(-1, FACTOR), axis=1)
        # np.mean(np.array(episode_epsilon).reshape(-1, FACTOR), axis=1)
        # np.mean(np.array(episode_rewards).reshape(-1, FACTOR), axis=1)
        # np.mean(np.array(episode_times).reshape(-1, FACTOR), axis=1)
        # # mean = (np.mean(np.array(episode_rewards).reshape(-1, 100), axis=1))

        # saveVec((np.array(episode_list)), 'episodes')
        # saveVec((np.array(episode_rewards)), 'rewards')
        # saveVec((np.array(episode_epsilon)), 'epsilon')
        # saveVec((np.array(episode_times)), 'time')

    # print('episode',episode_list,'reward', episode_rewards,'epsilon', episode_epsilon,'time', episode_time)
    #     if episode % 100:
    #         print('Episodio:', episode, 'Reward:', episode_reward)
    # np.mean(np.array(episode_rewards).reshape(-1, 100), axis=1)
    # print(np.mean(np.array(episode_rewards).reshape(-1, 100), axis=1))
    export(map, agent, Q)
    exp_rewards.append(episode_rewards)
    # saveVec((np.array(episode_times)), 'time')
    # plot_descripition = 'Rewards '+str(TEXT)+'_Entry_'+str(ALPHA)+'Gamma_'+str(EPSILON)+'Epsilon'
    #
    # plots(episode_list, episode_rewards, plot_descripition, 'blue', legenda='Reward')
    # plots(episode_list, episode_epsilon, 'Epislon Decay', 'red', legenda='Epislon Decay')



    # print(Q)


if __name__ == '__main__':

    # ALPHA_RANGE = [0.2, 0.3, 0.4]
    # FILE = ['entry00.txt', 'entry01.txt', 'entry02.txt', 'entry03.txt']
    FILE = ['entry02.txt', 'entry03.txt']
    # EPSILON_LIST = [0.85,  0.9, 0.95]
    # EPISODES_LIST = [10000, 20000, 30000]

    # for file in FILE:
    #     for ep in EPISODES_LIST:
    #         for a in ALPHA_RANGE:
    #             for epison_ in EPSILON_LIST:

    if len(sys.argv) == 5:
        try:
            txt, ALPHA, EPSILON, EPISODES = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
            exp_rewards = []
            run(txt, exp_rewards)
        except:
            raise Exception
        #
        # # DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        # # FOLDER2SAVE = str(DIR_PATH + '/results/' + str(txt) + 'world_'+str(EPISODES) + 'ep_' + str(GAMMA) + 'gamma_'
        # #                   + str(ALPHA) + 'alpha_' + str(EPSILON) + 'epsilon')
        # try:
        #     # dirPath = os.path.dirname(os.path.realpath(__file__))
        #     # path = str(FOLDER2SAVE)
        #     # os.mkdir(path)
        # except:
        #     pass

        # exp_rewards = []
        # run(txt, exp_rewards)

    if len(sys.argv) > 5:
        #Plot Results#

        exp_rewards = []

        fig, ax = plt.subplots()
        for j, f in enumerate(FILE):
            ALPHA, EPSILON, EPISODES = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
            # DIR_PATH = os.path.dirname(os.path.realpath(__file__))
            # FOLDER2SAVE = str(
            #     DIR_PATH + '/results/' + str(f) + 'world_' + str(EPISODES) + 'ep_' + str(GAMMA) + 'gamma_'
            #     + str(ALPHA) + 'alpha_' + str(EPSILON) + 'epsilon')
            for i in tqdm.trange(int(sys.argv[5])):
                run(f, exp_rewards)
            results = np.mean(np.array(exp_rewards), axis=0)
            std = np.std(np.array(exp_rewards), axis=0)
            results_f = gaussian_filter1d(results, 5)
            std_f = gaussian_filter1d(std, 5)
            ax.plot(np.arange(results.shape[0]), results_f, label=f)
            ax.fill_between(np.arange(results.shape[0]), results_f - np.sqrt(std_f), results_f + np.sqrt(std_f), alpha=0.10)
        ax.legend()
        plt.xlabel('Episódios')
        plt.ylabel('Reward Acumulado')
        plt.title(r"$\alpha = " + sys.argv[2] + r"$, $\epsilon = " + sys.argv[3] + r"$")
        plt.savefig('results_'+'alfa_' + sys.argv[2] + '_ep_' + sys.argv[3] + '.png')

