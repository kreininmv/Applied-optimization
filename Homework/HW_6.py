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
                    for j in random.sample(len(w), self._batch):
                        grad_j = self._X_train[j] @ w - self._b[j]
                        w[j] = w[j] - lr(w, self._i) * grad_j
                else:
                    for l in range(self._batch):
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
    
    def choose_opt_method(self, name):
        if (name == "SGD"):
            return self.__stochastic_gradient_descent
        if (name == "SGDB"):
            return self.__stochastic_gradient_descent_batch
        if (name == "SGDV"):
            return self.__stochastic_gradient_descent_vector
        if (name == "SCGD"):
            return self.__gradient_descent_coordinates
        if (name == "SCGD"):
            return self.__gradient_descent_coordinates_batch
        return self.__gradient_descent
    
    def get_name(self): return self._label



#######################################################################################################
################################         My logistic regression        ################################
#######################################################################################################
class MyLogisticRegression:
    def __init__(self, fit_intercept=False,  iter=10, l2=False, step=50, 
        l2_coef=1, name='default', eps=0.25, batch_size=200, method = "GD", 
        test=False, beta1 = 0.999, beta2=0.9, reg2 = 0.05):
        self.fit_intercept  = fit_intercept
        self._iter          = iter
        self._l2            = l2
        self._step          = step
        self._l2_coef       = l2_coef
        self._name          = name
        self._eps           = eps
        self._batch_size    = batch_size
        self._method        = method
        self._test          = test
        self._beta1         = beta1
        self._beta2         = beta2
        self._reg2          = reg2
        return

    def fit_test(self, X_train, y_train, X_test, y_test):

        X_train, w0 = self.__add_constant_column(X_train)
        X_test, _ = self.__add_constant_column(X_test)
        
        # Находим константу липшица для подбора learning rate
        hessian = np.zeros((X_train.shape[1], X_train.shape[1]))

        if self._name == 'GD':
            for x in X_train[:]:
                hessian = hessian + 1/(4*X_train.shape[0])  * np.outer(x, x)
            L = np.linalg.norm(hessian, 2)
        else:
            L_max = -1
            for x in X_train[:]:
                L = np.linalg.norm(1/4*np.outer(x, x))
                L_max = max(L, L_max)
            L = L_max
        
        print(L)
        
        #SAGA 1/6L
        #SVRG 1/6L
        #SARAH 1/2L
        lr_func = lambda w: 1.0/ L
        grad_function = self.__choose_gradient(self._l2)
        to_seconds = lambda t: t.microseconds * 1e-6 + t.seconds
        error_criterion = lambda x, y, w: np.linalg.norm(grad_function(x, y, w), 2)
        opt = self.choose_opt_method()
        
        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._l2_coef    = L/1000
        self._time       = []
        time_start = dt.now()
        num_batches = X_train.shape[0] // self._batch_size
        
        
        
        
        #loop = tqdm(range(int(self._iter)), total=int(self._iter), leave=False)
        for k in loop:
        
            error = error_criterion(X_train, y_train, self._w)
                
            self._time.append(to_seconds(dt.now() - time_start))
            self._errors.append(error) 
            self._accuracies.append(accuracy_score(y_test, self.predict(X_test)))
                
            if (self._test):
                print(f"{j + k*num_batches}: error({error}), accuracy_test({self._accuracies[-1]}), accuracy_batch({accuracy_score(y_batch, self.predict(X_batch))}), accuracy_train({accuracy_score(y_train, self.predict(X_train))})")

            loop.set_description(f"Method({self._name}) Epoch[{k}/{self._iter}], [error=%.4f, acc test=%3.1f, acc train=%3.1f]" % (error, 100.*self._accuracies[-1], 100.*accuracy_score(y_train, self.predict(X_train))))

            self._w = opt(self.__function, grad_function, self._w,
                              lr_func, error_criterion, X_batch, y_batch)
                
                
        return
    
    def _append_errors_accuracy(self):
        error = error_criterion(X_train, y_train, self._w)
                
            self._time.append(to_seconds(dt.now() - time_start))
            self._errors.append(error) 
            self._accuracies.append(accuracy_score(y_test, self.predict(X_test)))

    def _generate_batches(self, X, y, batch_size):
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

    def __gradient_descent(self, f, grad_f, w0, 
                           lr, error_criterion, X, y):
        """
        Градиентный спуск.
        """
    
        w = w0
        for k in range(self._step):
            w = w - lr(w) * grad_f(X, y, w)
            
            if (k % 30 == 0):
                error = error_criterion(X, y, w)
                if (error < self._eps):
                    return w
            
        return w
    

    def __function(self, x, y, w):
        sum = 0
        n = x.shape[0]

        for i in range(len(y)):     
            sum = sum + 1/n * np.log(1 + np.exp(-y[i] * x[i, :] @ w)) 
        
        return sum

    def __grad_function(self, x, y, w):
        sum = np.zeros(w.shape)
        n = x.shape[0]
        
        for i in range(len(y)):            
            up = y[i] * x[i] * np.exp(-y[i] * w * x[i])
            down = n * (1 + np.exp(-y[i] * w * x[i]))

            sum = sum  - up/down

        return sum

    def __add_constant_column(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            w0 = np.random.rand(k+1)
        else:
            X_train = X
            w0 = np.random.rand(k)
        
        return X_train, w0
    
    def __choose_gradient(self, l2):
        if l2:
            grad_l2 = lambda x, y, w: self.__grad_function(x, y, w) + self._l2_coef * w
            return grad_l2
        
        return self.__grad_function
    
    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time
    
    def get_name(self): return self._name
    
    def get_step(self): return self._step

    def choose_opt_method(self):
        if self._method == "SGD":
            return self.__stochastic_gradient_descent
        if self._method == "SAGA":
            return self.__SAGA
        if self._method == "SVRG":
            return self.__SVRG
        if self._method == "SARAH":
            return self.__SARAH
    
        return self.__gradient_descent
    
    def predict(self, X):
        y_pred = self.prob(X)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = -1

        return y_pred

    def calc_accuracy(self, X, y):
        count = 0
        X_train, c = self.__add_constant_column(X)
        for i in range(y.size):
            if X_train[i].dot(self._w) * y[i] > 0:
                count += 1
        return count / y.size

    def prob(self, X):
        X_train, c = self.__add_constant_column(X)
        
        y_prob = 1/(1 + np.exp(-X_train @ self._w))
    
        return y_prob