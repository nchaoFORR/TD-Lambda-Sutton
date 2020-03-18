import pandas as pd
import numpy as np
import random
import math
import copy

IDEAL_PREDS = [1/6, 1/3, 1/2, 2/3, 5/6]


def RMSE(ideal_preds, preds):
    return min(99, np.sqrt(np.mean(((ideal_preds - preds)**2))))


class RandomWalk:
    def __init__(self, size=5):
        self.size = size-1
        self.state = math.floor(self.size/2)
        self.X = [self._gen_obs(self.state)]

    def _reset(self):
        self.state = math.ceil(self.size/2)
        self.X = [self._gen_obs(self.state)]

    def _gen_obs(self, state):
        return [1 if i == state else 0 for i in range(self.size+1)]

    def _sequence(self):
        self._reset()
        while True:
            roll = np.random.uniform()
            if roll >= 0.5:
                if self.state == self.size:
                    return self.X
                else:
                    self.X.append(self._gen_obs(self.state+1))
                    self.state += 1
            else:
                if self.state == 0:
                    return self.X
                else:
                    self.X.append(self._gen_obs(self.state-1))
                    self.state -= 1
        return self.X

    def _gen_sequences(self, size=10):
        return [self._sequence() for _ in range(size)]

    def gen_trainset(self, size=100):
        return np.array([self._gen_sequences() for _ in range(size)])


class TDLambda:
    def __init__(self):
        self.lamb = None
        self.lr = None
        self.conv_tol = None
        self.max_iter = None
        self.w = None

    def reset_weights(self, size=5):
        self.w = np.array([0.5 for _ in range(size)])

    # policy function gradient
    def _grad_tail(self, s, explored, lamb):
        e = np.sum([lamb**(len(explored)-1-k)*np.array(s)
                    for k, s in enumerate(explored)], axis=None)
        return e

    # run through a sequence, accumulating a gradient
    # to add at the end of the sequence
    def _sequence(self, sequence, lamb, lr):
        w_delts = np.zeros(len(self.w))
        explored = []
        for i, s in enumerate(sequence):
            explored.append(s)
            if i+1 == len(sequence):
                e = np.sum([lamb**(len(explored)-1-k)*np.array(s)
                            for k, s in enumerate(explored)], axis=None)
                if s[0] == 1:
                    z = 0.0
                else:
                    z = 1.0
                w_delts += [lr*(z - np.dot(self.w, s))*e if i == 1 else 0
                            for i in s]
                break
            s_next = sequence[i+1]
            p_next = np.dot(self.w, np.transpose(s_next))
            p = np.dot(self.w, np.transpose(s))
            td = (p_next - p)
            grad_tail = self._grad_tail(s, explored, lamb)
            w_delts += [lr*td*grad_tail if i == 1 else 0 for i in s]
        self.w += w_delts

    # Fit a model against a batch of 10 sequences until
    # convergence
    def fit_batch(self, seq_batch, lamb, lr=0.3,
                  conv_tol=0.0001, max_iter=300, reset_w=True):
        self.lamb = lamb
        self.lr = lr
        self.conv_tol = conv_tol
        self.max_iter = max_iter
        if reset_w:
            self.w = np.array([0.5 for _ in range(5)])
        i = 0
        conv = 99
        while conv >= self.conv_tol and i < self.max_iter:
            seq_idx = np.random.randint(0, len(seq_batch)-1)
            prev_w = copy.copy(self.w)
            self._sequence(seq_batch[seq_idx], lamb, lr)
            conv = RMSE(prev_w, self.w)
            i += 1
