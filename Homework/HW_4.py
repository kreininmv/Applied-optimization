import numpy as np
import scipy.linalg as la
from sklearn.metrics import accuracy_score
from datetime import datetime as dt


class MyMirrorRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        return
 
    def error_criterion(self, w, X):
        grad = X.T@w
        return grad.T@w - np.amin(grad)
    
    def fit(self, X, eps = 5*10e-3, iter = 10 ** 3, step=10 ** 3, wolf=False):
        """
        Функция подбора параметров линейной модели для квадратичной функции потерь.

        Input: 
        - X     : матрица объектов
        - y     : вектор ответов
        - eps   : величина ошибки, до которой будем сходится
        - iter  : количество шагов, которое сделает алгоритм. Кол-во итераций = iter * step
        - l2    : добавлять или нет L2 регуляризацию
        - step  : количество итераций, которое работает градиентный спуск перед тем, как записать ошибку
        Returns: none.
        """
        
        # Находим константу липшица для подбора learning rate
        L = np.linalg.norm(X, 2)
        w0 = np.ones(X.shape[0])*1/X.shape[0]

        lr_func = lambda X, w: 1.0/L
        function = lambda w: w@X@w
        grad_function = lambda w: X.T@w
        error_criterion = self.error_criterion
        to_seconds = lambda t: t.microseconds * 1e-6 + t.seconds
        opt = self.__choose_opt_method(wolf)
        self._iter       = iter
        self._step       = step
        self._errors     = []
        self._w          = w0
        self._time       = []
        
        time_start = dt.now()

        for i in range(int(iter)):
            self._i = i
            self._w = opt(function, grad_function, self._w,
                        lr_func, step, error_criterion, 
                        X, eps)
            error = error_criterion(self._w, X)
            self._time.append(to_seconds(dt.now() - time_start))
            self._errors.append(error) 
            
            #print("%d: %f" %(i, error))
            
            if (error < eps):
                break
        
        return self


    def predict(self, X):
        """
        Функция предсказания по признакам, уже натренированной модели.
        Input: 
        - X : объекты, по которым будем предксазывать

        Returns: предсказания на основе весов, ранее обученной линейной модели.
        """        

        return self._w@X@self._w

    def __gradient_descent_mirror(self, f, grad_f, w0, 
                         lr, iter, error_criterion, 
                         X, eps):
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

        for k in range(iter):
            tmp = np.exp(-lr(X, w) * grad_f(w))
            w = w * tmp/(w*tmp).sum()

            if (k % 30 == 0):
                error = error_criterion(w, X)
                if (error < eps):
                    return w
         
        return w
    
    def __gradient_descent_wolf(self, f, grad_f, w0, 
                         lr, iter, error_criterion, 
                         X, eps):
        """
        Это градиентный спуск wolf.
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

        for k in range(iter):
            s = np.zeros(w.shape[0])
            s[np.argmin(grad_f(w))] = 1
            w = w + 2.0/(k + self._i * self._iter + 2)*(s - w)
            
            if (k % 30 == 0):
                error = error_criterion(w, X)
                if (error < eps):
                    return w
        return w

    def __choose_opt_method(self, wolf):
        if wolf:
            return self.__gradient_descent_wolf
        return __gradient_descent_mirror

    def get_errors(self):
        return self._errors

    def get_weights(self):
        return self._w

    def get_time(self):
        return self._time