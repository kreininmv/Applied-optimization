#@title Класс логистической регрессии
import numpy as np
import scipy.linalg as la
import random
from sklearn.metrics import accuracy_score
from datetime import datetime as dt
from tqdm import tqdm

class MyLinearRegression:
    def __init__(self, fit_intercept=False, eps=5*1e-3, iter =10 ** 3, batch=False, batch_size=50, decreasing_lr=False, name="GD", sigma=10, lr_func = None, label="None", dependent=False):
        """
        Функция инициализации функции потерь.
        Input: 
        - fit_intercept : добавлять bias столбец или нет
        - X             : матрица объектов
        - y             : вектор ответов
        - eps           : величина ошибки, до которой будем сходится
        - iter          : количество шагов, которое сделает алгоритм. Кол-во итераций = iter * step
        - step          : количество итераций, которое работает градиентный спуск перед тем, как записать ошибку
        Returns: none.
        """
        self.fit_intercept = fit_intercept
        self._iter         = iter
        self._batch        = batch
        self._eps          = eps
        self._opt          = self.choose_opt_method(name)
        self._name         = name
        self._batch_size   = batch_size
        self._decreasing_lr = decreasing_lr
        self._sigma        = sigma
        self._lr_func      = lr_func 
        self._label        = label
        self._dependent    = dependent

        return

    def fit(self, X, b):
        """
        Функция подбора параметров линейной модели для квадратичной функции потерь.
        Input: 
        - X     : матрица объектов
        - y     : вектор ответов
        Returns: none.
        """

        # Добавляем ещё столбец для константы
        X_train, w0 = self.__add_constant_column(X)
        
        # Create additional functions
        if self._lr_func is None:
            L = np.linalg.norm(X_train, 2)
            if self._decreasing_lr:
                self._lr_func = lambda w, i: 1.0/(L * i)
            else:
                self._lr_func = lambda w, i: 1.0/L
        
        self._grad_function = lambda w: w @ X_train - b
        self._error_criterion = lambda w: np.linalg.norm(self._grad_function(w), 2)
        self._function = lambda w: w@X_train@w - b@w
        self._to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
        self._X_train = X_train
        self._b = b

        # Initialize variables
        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._time       = []
        self._i          = 0

        self._w = self._opt(self._function, self._grad_function, self._w, self._lr_func)
        
        return self
    
    def predict(self, X, b):
        """
        Функция предсказания по признакам, уже натренированной модели.
        """        
        X_train, _ = self.__add_constant_column(X)
        y_pred = self._w @ X_train @ self._w - b @ self._w

        return y_pred

    def __stochastic_gradient_descent(self, f, grad_f, w0, lr):
        """
        Стохастический градиентный спуск, по вектору.
        """
        time_start = dt.now()
        w = w0
        for k in range(self._iter):
            self._i += 1
            vec = np.ones_like(w)
            w = w - lr(w, self._i) * (grad_f(w) + self._sigma * np.random.randn(1) * vec)
            
            self._append_errors(w, time_start)
            if (self._errors[-1] < self._eps):
                return w
            
        return w

    def __stochastic_gradient_descent_vector(self, f, grad_f, w0, lr):    
        """
        Стохастический градиентный спуск, с разной стохастикой по всем координатам.
        """
        time_start = dt.now()
        w = w0
        
        for k in range(self._iter):
            self._i += 1
            # Algorithm
            w = w - lr(w, self._i) * (grad_f(w) + self._sigma * np.random.randn(w.shape[0]))

            # Tecnichal staff 
            self._append_errors(w, time_start)
            if (self._errors[-1] < self._eps):
                return w
            
        return w

    def __stochastic_gradient_descent_batch(self, f, grad_f, w0, lr):
        """
        Стохастический градиентный спуск, с одинаковой стохастикой по всем координатам.
        """
        time_start = dt.now()
        w = w0
        for k in range(self._iter):
            self._i += 1
            xi = self._sigma / self._batch_size * np.random.randn(self._batch_size).sum()
    
            w = w - lr(w, self._i) * (grad_f(w) + xi * np.ones_like(w))
            
            # Tecnichal staff 
            
            self._append_errors(w, time_start)
            if (self._errors[-1] < self._eps):
                return w
            
        return w

    def __gradient_descent(self, f, grad_f, w0, lr):
        """
        Градиентный спуск.
        """
        time_start = dt.now()
        w = w0
        for k in range(self._iter):
            self._i += 1
            w = w - lr(w, self._i) * grad_f(w)
            
            self._append_errors(w, time_start)
            if (self._errors[-1] < self._eps):
                return w
            
        return w
    
    def __gradient_descent_coordinates(self, f, grad_f, w0, lr):
        """
        Покоординантный градиентный спуск.
        """
        time_start = dt.now()
        w = w0
        
        for k in range(self._iter):
            for r in range(len(w)):
                j = np.random.randint(len(w))
                self._i += 1
                grad_j = self._X_train[j] @ w - self._b[j]
                w[j] = w[j] - lr(w, self._i) * grad_j
            
            self._append_errors(w, time_start)
            if (self._errors[-1] < self._eps):
                return w

        return w
    
    def __gradient_descent_coordinates_batch(self, f, grad_f, w0, lr):
        """
        Покоординантный градиентный спуск.
        """
        time_start = dt.now()
        w = w0
    
        for k in range(self._iter):
            for r in range(len(w)):
                self._i += 1
                
                if self._dependent:
                    
                    for j in random.sample(range(len(w)), self._batch_size):
                        tmp = lr(w, self._i)
                        grad_j = self._X_train[j] @ w - self._b[j]
                        w[j] = w[j] - lr(w, self._i) * grad_j
                        
                else:
                    for l in range(self._batch_size):
                        j = np.random.randint(len(w))
                        grad_j = self._X_train[j] @ w - self._b[j]
                        w[j] = w[j] - lr(w, self._i) * grad_j
                
            
            self._append_errors(w, time_start)
            if (self._errors[-1] < self._eps):
                return w

        return w

    def __add_constant_column(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            w0 = np.random.rand(k+1)
        else:
            X_train = X
            w0 = np.random.rand(k)
        
        return X_train, w0
    
    def _append_errors(self, w, time_start):
        # Tecnichal staff 
        error = self._error_criterion(w)
                
        self._time.append(self._to_seconds(dt.now() - time_start))
        self._errors.append(error) 
        

    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time
    
    def __SEGA(self, grad_f, w0, lr):
        """
        Покоординантный градиентный спуск.
        """
        time_start = dt.now()
        w = w0
        h = grad_f(w)

        for k in range(self._iter):
            for r in range(len(w)):
                j = np.random.randint(len(w))
                self._i += 1
                if self._mush:
                    h[j] = 2/self.X_train.shape[0] * self._X_train.T[j, :] @ (self._X_train @ w - self._b)
                else:
                    h[j] = self._X_train[j] @ w - self._b[j]
                w = w - lr(w, self._i) * h
            
            self._append_errors(w, time_start)
            if (self._errors[-1] < self._eps):
                return w

        return w
    
    def choose_opt_method(self, name):
        if (name == "SGD"):
            return self.__stochastic_gradient_descent
        if (name == "SGDB"):
            return self.__stochastic_gradient_descent_batch
        if (name == "SGDV"):
            return self.__stochastic_gradient_descent_vector
        if (name == "SCGD"):
            return self.__gradient_descent_coordinates
        if (name == "SCGDB"):
            return self.__gradient_descent_coordinates_batch
        if (name == "SEGA"):
            return self.__SEGA
        return self.__gradient_descent

    def get_name(self): return self._label




class Mushrooms:
    
    def __init__(self, fit_intercept=False, eps=5*1e-3, iter =10 ** 3, batch=False, batch_size=50, decreasing_lr=False, name="GD", lr_func = None, label="None", dependent=False, l2_coef=0):
        self.fit_intercept = fit_intercept
        self._iter         = iter
        self._batch        = batch
        self._eps          = eps
        self._opt          = self.choose_opt_method(name)
        self._name         = name
        self._batch_size   = batch_size
        self._decreasing_lr = decreasing_lr
        self._lr_func      = lr_func 
        self._label        = label
        self._dependent    = dependent
        self._l2_coef      = l2_coef
    def __function(self, X, y, w, n):
        return 1/n * la.norm(w @ X.T - y, 2) ** 2
    
    def __grad_function(self, X, y, w, n):
        return 1/n * 2 * X.T @ (w @ X.T - y).T 

    def __add_constant_column(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            w0 = np.random.rand(k+1)
        else:
            X_train = X
            w0 = np.random.rand(k)
        
        return X_train, w0
    
    def fit(self, X, y, X_test, y_test):
        n, k = X.shape
        wo = np.array([])
        X_train, w0 = self.__add_constant_column(X)
        self._X_train = X_train
        self._y_train = y
        self._X_test = X_test
        self._y_test = y_test

        if self._lr_func is None:
            hessian = 2 / n * X_train.T @ X_train
            wb, vb = np.linalg.eigh(hessian)
            self._lr_func = lambda w, x: 1/wb[-1]
        
        self._error_criterion = lambda X, y, w, n: np.linalg.norm(self.__grad_function(X, y, w, n), 2)
        self._grad_function = lambda w: 2/n * X_train.T @ (X_train @ w - y)
        self._error_criterion = lambda w: np.linalg.norm(self._grad_function(w), 2)
        self._to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
        # Initialize variables
        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._time       = []
        self._i          = 0

        self._w = self._opt(self.__function, self._grad_function, self._w, self._lr_func)
        return 

    def predict(self, X):
        return self._w @ X.T

    def get_weights(self): return self._w
    
    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time

    def get_name(self): return self._label
  
    def __gradient_descent(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        n = self._X_train.shape[0]
        for k in range(self._iter):
            self._i += 1
            self._w = self._w - lr(0, self._i) * self.__grad_function(self._X_train, self._y_train, self._w, n)

            self._append_errors(self._w, time_start)
            
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w
    
    def __gradient_descent_coordinates(self, f, grad_f, w0, lr):
        """
        Координатный градиентный спуск
        """
        time_start = dt.now()
        self._w = w0
        for k in range(self._iter):
            for r in range(len(self._w)):
                j = np.random.randint(len(self._w))
                self._i += 1
                grad_j = 2/self._X_train.shape[0] * self._X_train.T[j, :] @ (self._X_train @ self._w - self._y_train)
                self._w[j] = (1-self._l2_coef*lr(self._w, self._i))*self._w[j] - lr(self._w, self._i) * grad_j
            
            self._append_errors(self._w, time_start)
            if (self._errors[-1] < self._eps):
                return self._w
        return self._w
    def __SEGA(self, f, grad_f, w0, lr):
        """
        SEGA
        """
        time_start = dt.now()
        self._w = w0
        h = grad_f(self._w)

        for k in range(self._iter):
            self._i += 1
            for j in random.sample(range(len(self._w)), self._batch_size):
                h[j] = 2/self._X_train.shape[0] * self._X_train.T[j, :] @ (self._X_train @ self._w - self._y_train)
            
            self._w = (1-self._l2_coef*lr(self._w, self._i))*self._w - lr(self._w, self._i) * h
            
            self._append_errors(self._w, time_start)
            if (self._errors[-1] < self._eps):
                return self._w

        return self._w

    def _append_errors(self, w, time_start):
        # Tecnichal staff 
        error = self._error_criterion(self._w)
                
        self._time.append(self._to_seconds(dt.now() - time_start))
        self._errors.append(error) 
        
        answer = self.predict(self._X_test)
        answer = np.sign(answer)
        #print(answer)
        self._accuracies.append(accuracy_score(self._y_test, answer))
    
    def choose_opt_method(self, name):
        if (name == "SCGD"):
            return self.__gradient_descent_coordinates
        if (name == "SEGA"):
            return self.__SEGA
        return self.__gradient_descent
