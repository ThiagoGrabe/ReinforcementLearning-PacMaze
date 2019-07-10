class Agent:
    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.avatar = 'A'

    def getNewPosition(self, direction):

        if direction == 'UP':
            self.x -= 1
            self.y = self.y

        if direction == 'DOWN':
            self.x += 1
            self.y = self.y

        if direction == 'RIGHT':
            self.x = self.x
            self.y += 1

        if direction == 'LEFT':
            self.x = self.x
            self.y -= 1

        return self.x, self.y

    def getPose(self):
        return self.x, self.y

    def getAvatar(self):
        return self.avatar