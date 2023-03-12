#@title Класс логистической регрессии
import numpy as np
import scipy.linalg as la
from sklearn.metrics import accuracy_score
from datetime import datetime as dt

class MyLinearRegression:
    def __init__(self, fit_intercept=False, eps = 5*10e-3, iter = 10 ** (6-3), step=1000, stochastic = False, name="GD", batch_size = 10, batch=False):
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
        self._step         = step
        self._eps          = eps
        self._opt          = self.choose_opt_method(stochastic, batch)
        self._name         = name
        self._batch_size   = batch_size
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
        lr_func = lambda w: 1.0/L
        grad_function = lambda w: w @ X_train - b
        self._error_criterion = lambda w: np.linalg.norm(grad_function(w), 2)
        to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
        function = lambda w: w@X_train@w - b@w
        

        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._time       = []
        time_start = dt.now()

        for i in range(int(self._iter)):
            self._i = i
            self._w = self._opt(function, grad_function, self._w, lr_func)
            
            error = self._error_criterion(self._w)
            self._time.append(to_seconds(dt.now() - time_start))
            self._errors.append(error) 
            
            #print("%d: %f" %(i, error))
            
            if (error < self._eps):
                break
        
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

    def __stochastic_gradient_descent(self, f, grad_f, w0, lr):
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
    
        w = w0
        for k in range(self._iter):
            w = w - lr(w) * (grad_f(w) + 100 * np.random.randn(w.shape[0]))
            
            if (k % 30 == 0):
                error = self._error_criterion(w)
                if (error < self._eps):
                    return w
            
        return w


    def __stochastic_gradient_descent_batch(self, f, grad_f, w0, lr):
        w = w0
        for k in range(self._iter):
            xi = 100 * np.random.randn(w.shape[0], self._batch_size)

            xi = xi.sum(axis=1) * 1/self._batch_size
            w = w - lr(w) * (grad_f(w) + xi)
            
            if (k % 30 == 0):
                error = self._error_criterion(w)
                if (error < self._eps):
                    return w
            
        return w

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
    
        w = w0
        prev_w = w
        for k in range(self._iter):
            w = w - lr(w) * grad_f(w)
            
            if (k % 30 == 0):
                error = self._error_criterion(w)
                if (error < self._eps):
                    return w
            
            prev_w = w
            
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
    
    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time
    
    def get_step(self): return self._step
    
    def choose_opt_method(self, stochastic, batch):
        if (stochastic):
            return self.__stochastic_gradient_descent
        if (batch):
            return self.__stochastic_gradient_descent_batch
        return self.__gradient_descent
    
    def get_name(self): return self._name



#########################################
#         My logistic regression        #
#########################################
class MyLogisticRegression:
    def __init__(self, fit_intercept=False,  iter = 10 ** (6-3), l2 = False, step=1000, 
        l2_coef = 1, name='default', eps = 5*10e-3, batch_size = 10):
        self.fit_intercept  = fit_intercept
        self._iter          = iter
        self._l2            = l2
        self._step          = step
        self._l2_coef       = l2_coef
        self._name          = name
        self._eps           = eps
        self._batch_size    = batch_size
        return

    def fit_test(self, X_train, y_train, X_test, y_test):

        X_train, w0 = self.__add_constant_column(X_train)
        X_test, _ = self.__add_constant_column(X_test)
        
        # Находим константу липшица для подбора learning rate
        hessian = np.zeros((X_train.shape[1], X_train.shape[1]))
        
        for x in X_train[:]:
            hessian = hessian + 1/(4*X_train.shape[0])  * np.outer(x, x)
        L = np.linalg.norm(hessian, 2)

        lr_func = lambda w: 1.0/ L
    
        grad_function = self.__choose_gradient(self._l2)
        to_seconds = lambda t: t.microseconds * 1e-6 + t.seconds
        error_criterion = lambda x, y, w: np.linalg.norm(grad_function(x, y, w), 2)
        
        
        opt = self.choose_opt_method()
        self._iter       = iter
        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._time       = []
        
        time_start = dt.now()

        num_batches = X_train.shape[0] // self._batch_size
        
        
        for k in range(int(self._iter)):
            for j in range(num_batches - 1):
                X_batch = X_train[j * self._batch_size: (j + 1) * self._batch_size, :]
                y_batch = y_train[j * self._batch_size: (j + 1) * self._batch_size]
            
                
                self._w = opt(self.__function, grad_function, self._w,
                                                lr_func, error_criterion, 
                                                X_batch, y_batch)
    
                
                error = error_criterion(X_batch, y_batch, self._w)
    
                self._time.append(to_seconds(dt.now() - time_start))
                self._errors.append(error) 
                self._accuracies.append(accuracy_score(y_batch, self.predict(X_batch)))
                
                #print("%d: %f" %(i, error))
                
                if (error < self._eps):
                    break
        
        return self

    def predict(self, X):
        """
        Функция предсказания по признакам, уже натренированной модели.
        Input: 
        - X : объекты, по которым будем предксазывать

        Returns: предсказания на основе весов, ранее обученной линейной модели.
        """        
        y_pred = self.prob(X)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = -1
        
        '''
        for i in range(len(y_pred)):
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = -1
            if (y_pred[i] >= 0.5):
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        '''
        #y_pred = np.sign(self._w @ X_train.T)

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

    def __gradient_descent(self, f, grad_f, w0, 
                         lr, iter, error_criterion, 
                         X, y, eps):
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
        - iter              : количество итераций.
        - error_criterion   : функция подсчета ошибки
        - X_train           : множество объектов (матрица фичей)
        - y                 : вектор ответов
        - eps               : величина ошибки, до которой будем сходится

        Returns:
        Наилучшую минимальную точку, которую удалось найти.
        """
    
        w = w0
        prev_w = w
        for k in range(iter):
            w = w - lr(X, y, w) * grad_f(X, y, w)
            
            if (k % 30 == 0):
                error = error_criterion(X, y, w, prev_w)
                if (error < eps):
                    return w
            
            prev_w = w
            
        return w


    def __function(self, x, y, w):
        """
        Целевая фукнция, которую минимизирует наша логистическая регрессия.

        Input: 
        - X   : матрица фичей
        - y   : вектор ответов
        - w   : параметры модели
        """
        sum = 0
        n = x.shape[0]

        for i in range(len(y)):     
            sum = sum + 1/n * np.log(1 - np.exp(-y[i] * x[i, :] @ w)) 
        
        return sum

    def __grad_function(self, x, y, w):
        sum = np.zeros(w.shape)
        n = x.shape[0]
        
        for i in range(len(y)):            
            sum = sum  - 1/n * y[i] * x[i] /(1 + np.exp(y[i] * w * x[i]))

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
            return lambda x, y, w: self.__grad_function(x, y, w) + 2 * self._l2_coef * w
        return self.__grad_function
    
    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time
    
    def get_name(self): return self._name
    
    def choose_opt_method(self):
        return self.__gradient_descent
    
    def proj(self, x0):
        rad = self.radius
        x = np.copy(x0)
        d = int (x0.shape[0]) + 1

        if (np.linalg.norm(x, 1) <= rad):
            lambd = 0
        else:
            x = np.absolute(x)
            x = np.append(x, 0)
            x = np.sort(x)

            x_first = np.flip(x)
            first = np.flip(np.cumsum(x_first))

            second = np.multiply( np.arange(d), x )

            third = x * (-d)

            forth = np.ones(d) * (-rad)

            g_deriv = 2 * ( first + second + third + forth )
            k  = np.min(np.where(g_deriv[1:] * g_deriv[:-1] <= 0))

            lambd = (first[k] - x[k] - rad) / (d - 1 - k) 

        y = np.zeros(d - 1)
        y[x0 <= -lambd] = x0[x0 <= -lambd] + lambd
        y[x0 >= lambd] = x0[x0 >= lambd] - lambd

        return y

