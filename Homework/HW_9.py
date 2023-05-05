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
    
    def broyden(self, w, H, k):
        grad_prev = self.grad_func(w, self.args)
        d = - H @ grad_prev
        lr = self.learning_rate(w, self.args)
        w, w_prev = w + lr * d, w
        grad = self.grad_func(w, self.args)
        s = w - w_prev
        y = grad - grad_prev
        q = s - H @ y
        mu = 1 / (q @ y)
        H = H + mu * np.array([q]).T @ np.array([q])
        
        return w, H
    
    def dfp(self, w, H, k):
        grad_prev =  self.grad_func(w, self.args)
        d = - H @ grad_prev
        lr = self.learning_rate(w, self.args)
        w, w_prev = w + lr * d, w
        grad = self.grad_func(w, self.args)
        s = np.array([w - w_prev]).T
        y = np.array([grad - grad_prev]).T
        
        H = H - (H @ y @ y.T @ H) / (y.T @ H @ y) + s @ s.T / (y.T @ s)

        return w, H, w_prev
    
    def bfgs(self, w, H, k):
        grad_prev = self.grad_func(w, self.args)
        
        d = - H @ grad_prev
        lr = self.learning_rate(w, self.args)
        w, w_prev = w + lr * d, w
        grad = self.grad_func(w, self.args)
        
        s = np.array([w - w_prev]).T
        y = np.array([grad - grad_prev]).T
        
        rho = 1 / (y.T @ s)
        V = np.eye(len(w)) - rho * y @ s.T
        H = V.T @ H @ V + rho * s @ s.T
        
        return w, H, w_prev
    

    def lbfgs(self, w, H, H0, ss, ys, k):
        grad_prev = self.grad_func(w, self.args)

        d = - H @ grad_prev
        lr = self.learning_rate(w, self.args)
        w, w_prev = w + lr * d, w
        grad = self.grad_func(w, self.args)

        s = w - w_prev
        y = grad - grad_prev
        m = self.args['m']
        
        for i in range(m - 1):
            ss[ : , i] = ss[ : , i + 1]
            ys[ : , i] = ys[ : , i + 1]
        ss[ : , m - 1] = s
        ys[ : , m - 1] = y

        H = H0
        for i in range(max(0, m - k - 1), m):
            s = np.array([ss[ : , i ]]).T
            y = np.array([ys[ : , i ]]).T
            rho = 1 / (y.T @ s)
            V = np.eye(len(w)) - rho * y @ s.T
            H = V.T @ H @ V + rho * s @ s.T
        return w, H, ss, ys

    def fit(self):
        to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
        time_start = dt.now()
        
        H = self.hessian_func(self.w, self.args)
        H0 = np.copy(H)
        ss = np.zeros((len(self.w), 3))
        ys = np.zeros((len(self.w), 3))

        for k in range(self.iter):
            w_prew = self.w
            if self.name == 'gd':
                self.w = self.gd(self.w, k)
            elif self.name == 'newton':
                self.w = self.newton(self.w, k)
            elif self.name == 'cubicnewton':
                self.w = self.cubic_newton(self.w, k)
            elif self.name == 'dfp':
                self.w, H, w_prew = self.dfp(self.w, H, k)
            elif self.name  == 'bfgs':
                self.w, H, w_prew = self.bfgs(self.w, H, k)
            elif self.name == 'lbfgs':
                self.w, H, ss, ys = self.lbfgs(self.w, H, H0, ss, ys, k)
            elif self.name == 'broyden':
                self.w, H = self.broyden(self.w, H, k)

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