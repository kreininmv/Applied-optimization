#@title Класс логистической регрессии
import numpy as np
import scipy.linalg as la
from sklearn.metrics import accuracy_score
from datetime import datetime as dt
from tqdm import tqdm

class MyLinearRegression:
    def __init__(self, fit_intercept=False, eps=5*1e-3, iter =10 ** 3, batch=False, batch_size=50, decreasing_lr=False, name="GD", sigma=10, lr_func = None, label="None"):
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
                grad_i = np.zeros_like(w)
                grad_i[j] = self._X_train[j] @ w - self._b[j]
                w = w - lr(w, self._i) * grad_i
            
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
    
    def choose_opt_method(self, name):
        if (name == "SGD"):
            return self.__stochastic_gradient_descent
        if (name == "SGDB"):
            return self.__stochastic_gradient_descent_batch
        if (name == "SGDV"):
            return self.__stochastic_gradient_descent_vector
        if (name == "GDC"):
            return self.__gradient_descent_coordinates
        return self.__gradient_descent
    
    def get_name(self): return self._label