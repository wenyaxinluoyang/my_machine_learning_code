
import numpy as np

class Distance:

    def dist(self, x, y):
        pass

class Euclidean(Distance):
    '''
    欧式距离(x1, y1), (x2, y2)
    dist = sqrt((x1-x2)^2 + (y1-y2)^2)
    '''

    def dist(self, x, y):
        d = np.sqrt(np.sum(np.square(x - y)))
        return d

class Manhattan(Distance):
    '''
    曼哈顿距离, (x1, y1), (x2, y2)
    dist = |x1 - x2| + |y1 - y2|
    '''
    def dist(self, x, y):
        d = np.sum(np.fabs(x - y))
        return d


class Minkowski(Distance):

    def dist(self, x, y):
        p = len(x)
        t = np.sum(np.power(np.fabs(x - y), p))
        d = pow(t, 1/p)
        return d

