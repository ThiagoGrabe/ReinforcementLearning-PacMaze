import numpy as np


class PacMazeWorld:
    def __init__(self, world):

        self.world = []
        self.world_reset = []
        self._world_ = world

        self.actions = ['RIGHT', 'LEFT', 'UP', 'DOWN']
        self.num_actions = len(self.actions)

        self.rows = None
        self.columns = None

        self.empty = '-'
        self.gost = '&'
        self.gold = '0'
        self.wall = '#'
        self.agent = 'A'

    def worldShape(self):
        with open(str(self._world_)) as file:
            for line in file:
                self.world_reset.append(list(map(str, line.split())))
        self.rows = int(self.world_reset[0][0])
        self.columns = int(self.world_reset[0][1])
        return self.rows, self.columns

    def reset(self):
        with open(str(self._world_)) as file:
            for line in file:
                self.world.append(list(map(str, line.split())))
        self.world = self.world[1:]
        for ix, array in enumerate(self.world):
            for jx, string in enumerate(array):
                self.world[ix][jx] = list(string)
        return self.world

    def getGoldPose(self):
        i, j = np.where(self.world == self.gold)
        return i, j

    def getGostPose(self):
        i, j = np.where(self.world == self.gost)
        return i, j

    def getWallPose(self):
        i_, j_ = np.where(self.world == self.wall)
        return i_, j_

    def getEmptyPose(self):
        i_, j_ = np.where(self.world == self.empty)
        return i_, j_

    def getPossibleStates(self):
        i_, j_ = np.where(self.world != self.wall)
        return i_, j_

    def getPossibleStatesTOPRINT(self):
        i_, j_ = np.where((self.world != self.wall) & (self.world != self.gold) & (self.world != self.gost))
        return i_, j_

    def getAgentState(self, i, j):
        return self.world[i][j]

    def getActions(self):
        return self.actions



