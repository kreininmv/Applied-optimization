import numpy as np
import scipy.linalg as la
import math
from math import atan, log, pi
from datetime import datetime as dt
from sklearn.metrics import accuracy_score

class Optimizer:
    def __init__(self, func, grad_func, hessian_func, w0, learning_rate, iter, args, name, label):
        self.func = func
        self.grad_func = grad_func
        self.w = w0
        self.iter = iter
        self.learning_rate = learning_rate
        self.args = args
        self.name = name
        self.label = label
        self.errors = []
        self.accuracy = []
        self.time = []
        self.v = self.grad_func(self.w, self.args)
        self.hessian_func = hessian_func
        self.args['w_prev'] = self.w

    def gd(self, w, k):
        lr = self.learning_rate(w, self.args)
        return w - lr * self.grad_func(w, self.args) 
    def predict(self, X):
        return np.sign(X @ self.w)
    
    def cubic_newton(self, x, k):
        M = 3/8*np.sqrt(3)
        D1 = (1/(1+x**2)) ** 2 + 2*M*x*atan(x)
        D2 = (1/(1+x**2)) ** 2 - 2*M*x*atan(x)
        f = lambda y: atan(x)*(y-x) + 0.5/(1+x**2)*(y-x)**2 + M/6*abs(y-x)**3
        
        y1 = x + (-1/(1+x**2) + np.sqrt(D1))/M
        y2 = x + (-1/(1+x**2) - np.sqrt(D1))/M
        y3 = x + (-1/(1+x**2) + np.sqrt(D2))/M
        y4 = x + (-1/(1+x**2) - np.sqrt(D2))/M

        if f(y1) > f(y2):
            y1 = y2
        if f(y1) > f(y3):
            y1 = y3
        if f(y1) > f(y4):
            return y4

        return y1

    def newton(self, w, k):
        lr = self.learning_rate(w, self.args)
        if self.args['test']:
            return w - lr*np.linalg.inv(self.hessian_func(w, self.args)) @ self.grad_func(w, self.args)
        return w - lr*self.grad_func(w, self.args) / self.hessian_func(w, self.args)
    
    def fit(self):
        to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
        time_start = dt.now()
        for k in range(self.iter):
            w_prew = self.w
            if self.name == 'gd':
                self.w = self.gd(self.w, k)
            elif self.name == 'newton':
                self.w = self.newton(self.w, k)
            elif self.name == 'cubicnewton':
                self.w = self.cubic_newton(self.w, k)
        
            self.args['w_prev'] = w_prew
            if self.args['test']:
                error = np.linalg.norm(self.grad_func(self.w, self.args), 2)
            else:
                error = abs(self.grad_func(self.w, self.args))

            self.time.append(to_seconds(dt.now() - time_start))
            self.errors.append(error)
            
            if self.args['test']:
                self.accuracy.append(accuracy_score(self.predict(self.args['X_test']), self.args['y_test']))
            
            if error < 1e-8:
                break