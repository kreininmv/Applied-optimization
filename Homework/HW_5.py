#@title Класс логистической регрессии
import numpy as np
import scipy.linalg as la
from sklearn.metrics import accuracy_score
from datetime import datetime as dt
class MyLinearRegression:
    def __init__(self, fit_intercept=False, eps=5*1e-3, iter =10 ** 3, step=1000, stochastic = False, stochastic_vector=False, batch=False, batch_size=50, decreasing_lr=False, name="GD", sigma=10):
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
        self._eps          = eps
        self._opt          = self.choose_opt_method(stochastic, batch, stochastic_vector)
        self._name         = name
        self._batch_size   = batch_size
        self._decreasing_lr = decreasing_lr
        self._sigma        = sigma
        self._to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds

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
               
        L = np.linalg.norm(X_train, 2)
        
        # Create additional functions
        
        if self._decreasing_lr:
            self._lr_func = lambda w, i: 1.0/(L * i)
        else:
            self._lr_func = lambda w, i: 1.0/L
        
        self._grad_function = lambda w: w @ X_train - b
        self._error_criterion = lambda w: np.linalg.norm(self._grad_function(w), 2)
        self._function = lambda w: w@X_train@w - b@w

        # Initialize variables
        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._time       = []
        self._i = 0
        self._w = self._opt(self._function, self._grad_function, self._w, self._lr_func)
        
        return self
    
    def predict(self, X, b):
        """
        Функция предсказания по признакам, уже натренированной модели.
        Input: 
        - X : объекты, по которым будем предксазывать

        Returns: предсказания на основе весов, ранее обученной линейной модели.
        """        
        X_train, _ = self.__add_constant_column(X)
        y_pred = self._w @ X_train @ self._w - b @ self._w

        return y_pred
    
    def _append_errors(self, w, time_start):
        # Tecnichal staff 
        error = self._error_criterion(w)
                
        self._time.append(self._to_seconds(dt.now() - time_start))
        self._errors.append(error) 

    def __stochastic_gradient_descent(self, f, grad_f, w0, lr):
        """
        Стохастический градиентный спуск
        """
        time_start = dt.now()
        self._w = w0
        
        for k in range(self._iter):
            self._i += 1
            vec = np.ones_like(self._w)
            self._w = self._w - lr(self._w, self._i) * (grad_f(self._w) + self._sigma * np.random.randn(1) * vec)
            
            self._append_errors(self._w, time_start)
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w

    def __stochastic_gradient_descent_vector(self, f, grad_f, w0, lr):    
        """
        Стохастический градиентный спуск со случайным вектором
        """
        
        time_start = dt.now()
        self._w = w0

        for k in range(self._iter):
            self._i += 1
            self._w = self._w - lr(self._w, self._i) * (grad_f(self._w) + self._sigma * np.random.randn(self._w.shape[0]))
            
            self._append_errors(self._w, time_start)
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w

    def __stochastic_gradient_descent_batch(self, f, grad_f, w0, lr):
        """
        Стохастический градиентный спуск по батчу
        """

        time_start = dt.now()
        self._w = w0
        for k in range(self._iter):
            self._i += 1
            xi = self._sigma / self._batch_size * np.random.randn(self._batch_size).sum()
    
            self._w = self._w - lr(self._w, self._i) * (grad_f(self._w) + xi * np.ones_like(self._w))
            
            self._append_errors(self._w, time_start)
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w

    def __gradient_descent(self, f, grad_f, w0, lr):
        """
        Это градиентный спуск.
        Он получает на вход целевую функцию, функцию градиента целевой функции, 
        начальную точку, функцию learning rate, количество итераций и 
        функцию подсчета ошибки. И применяетметод градиентного спуска.

        Inputs:
        - f                 : целевая функция, минимум которой мы хотим найти.
        - grad_f            : функция градиента целевой функции.
        - x0                : начальная точка.
        - lr                : функция learning rate.

        Returns:
        Наилучшую минимальную точку, которую удалось найти.
        """
        time_start = dt.now()
        
        self._w = w0
        for k in range(self._iter):
            self._i += 1
            self._w = self._w - lr(self._w, self._i) * grad_f(self._w)
            
            self._append_errors(self._w, time_start)
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w
    
    def __add_constant_column(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            w0 = np.random.rand(k+1)
        else:
            X_train = X
            w0 = np.random.rand(k)
        
        return X_train, w0
    
    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time
    
    def choose_opt_method(self, stochastic, batch, stochastic_vector):
        if (stochastic):
            return self.__stochastic_gradient_descent
        if (batch):
            return self.__stochastic_gradient_descent_batch
        if (stochastic_vector):
            return self.__stochastic_gradient_descent_vector
        return self.__gradient_descent
    def get_name(self): return self._name



#######################################################################################################
################################         My logistic regression        ################################
#######################################################################################################
class MyLogisticRegression:
    def __init__(self, fit_intercept=False, eps=5*1e-3, iter =10 ** 3, batch=False, batch_size=50, decreasing_lr=False, name="GD", lr_func = None, label="None", dependent=False, l2_coef=0, betas = [0.999, 0.99]):
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
        self._betas        = betas
        self._grad_function = self.__grad_function

        self._to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
    
    def __function(self, w):
        sum = 0
        n = self._X_train.shape[0]

        for i in range(len(self._y_train)):     
            sum = sum + 1/n * np.log(1 + np.exp(-self._y_train[i] * self._X_train[i, :] @ w)) 
        
        return sum
    
    def __grad_function(self, w):
        sum = np.zeros(w.shape)
        n = self._X_train.shape[0]
        
        for i in range(len(self._y_train)):            
            up = self._y_train[i] * self._X_train[i] * np.exp(-self._y_train[i] * w * self._X_train[i])
            down = n * (1 + np.exp(-self._y_train[i] * w * self._X_train[i]))
            sum = sum  - up/down

        return sum
    
    def __grad_function_part(self, w, X, y):
        sum = np.zeros(w.shape)
        n = X.shape[0]
        
        for i in range(len(y)):            
            up = y[i] * self._X_train[i] * np.exp(-y[i] * w * y[i])
            down = n * (1 + np.exp(-y[i] * w * X[i]))
            sum = sum  - up/down

        return sum

    def __add_constant_column(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            #w0 = np.random.rand(k+1)
            w0 = np.ones(k+1)
        else:
            X_train = X
            #w0 = np.random.rand(k)
            w0 = np.ones(k)
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
            self._lr_func = lambda w: 1/wb[-1]
        
        self._error_criterion = lambda w: np.linalg.norm(self.__grad_function(w), 2)
        
        # Initialize variables
        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._time       = []
        self._i          = 0

        self._w = self._opt(self.__function, self._grad_function, self._w, self._lr_func)
        return 
  
    def __gradient_descent(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        for k in range(self._iter):
            self._i += 1
            self._w = self._w - lr(0) * grad_f(self._w)
            self._append_errors(self._w, time_start)
            
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w
    
    def _get_batch(self, j):
        if j == self._n_batches - 1:
            X = self._X_train[j*self._batch_size : ]
            y= self._y_train[j*self._batch_size : ]
        else:
            X = self._X_train[j*self._batch_size : (j + 1) * self._batch_size]
            y = self._y_train[j*self._batch_size : (j + 1) * self._batch_size]
        return X, y
    
    def __SGD(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        
        self._n_batches = self._X_train.shape[0] // self._batch_size
        generator = self._generate_batches(self._X_train, self._y_train, self._batch_size)

        for k in range(self._iter):
            X, y = next(generator)

            self._i += 1
            self._w = self._w - lr(0) * self.__grad_function_part(self._w, X, y)
            self._append_errors(self._w, time_start)
            
            if (self._errors[-1] < self._eps):
                return self._w
        return self._w
    
    def __SAGA(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        generator = self._generate_batches(self._X_train, self._y_train, self._batch_size)
        
        n_batches = self._X_train.shape[0] // self._batch_size
        sum = np.zeros_like(w0)
        phi = np.zeros((n_batches, len(w0)))

        for _ in range(n_batches):
            X, y = next(generator)
            phi[j] = self.__grad_function_part(self._w, X, y)
            sum += phi[j]

        for _ in range(self._iter):
            X, y = next(generator)
            self._i += 1
            j = np.random.randint()


    
    def _generate_batches(self, X, y, batch_size):
        for i in range(1000):
            X = np.array(X)
            y = np.array(y)
            perm = np.random.permutation(len(X))
            perm = perm[:((len(X) // batch_size) * batch_size)]
            X_batch = []
            y_batch = []

            for batch_start in np.hsplit(perm, len(X) // batch_size): 
                for i in range(len(batch_start)):
                    X_batch.append(X[batch_start[i], :])
                    y_batch.append(y[batch_start[i]])
                yield (np.array(X_batch), np.array(y_batch))
                X_batch = []
                y_batch = []

    def predict(self, X):
        return X @ self._w

    def get_weights(self): return self._w
    
    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time

    def get_name(self): return self._label

    def _append_errors(self, w, time_start):
        # Tecnichal staff 
        error = self._error_criterion(self._w)
                
        self._time.append(self._to_seconds(dt.now() - time_start))
        self._errors.append(error) 
        
        answer = self.predict(self._X_test)
        answer = np.sign(answer)
    
        self._accuracies.append(accuracy_score(self._y_test, answer))
    
    def choose_opt_method(self, name):
        if (name == 'SGD'):
            return self.__SGD
        return self.__gradient_descent

