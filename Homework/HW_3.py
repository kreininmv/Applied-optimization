#@title Класс логистической регрессии
import numpy as np
import scipy.linalg as la
from sklearn.metrics import accuracy_score
from datetime import datetime as dt

class MyLogisticRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        return
 
    def fit(self, X, y, eps = 5*10e-3, iter = 10 ** (6-3), l2 = False, step=1000, 
        l2_coef = 1, heavy_ball = False, nesterov_moment = False, beta = 0.1, dynamic_beta=0):
        """
        Функция подбора параметров линейной модели для квадратичной функции потерь.

        Input: 
        - X     : матрица объектов
        - y     : вектор ответов
        - eps   : величина ошибки, до которой будем сходится
        - iter  : количество шагов, которое сделает алгоритм. Кол-во итераций = iter * step
        - l2    : добавлять или нет L2 регуляризацию
        - step  : количество итераций, которое работает градиентный спуск перед тем, как записать ошибку
        - alpha : гиперпараметр L2 регуляризации
        Returns: none.
        """

        # Добавляем ещё столбец для константы
        X_train, w0 = self.__add_constant_column(X)
        
        # Находим константу липшица для подбора learning rate
        hessian = np.zeros((X_train.shape[1], X_train.shape[1]))
        
        for x in X_train[:]:
            hessian = hessian + 1/(4*X_train.shape[0])  * np.outer(x, x)
        L = np.linalg.norm(hessian, 2)

        lr_func = lambda X, y, w, k: 1.0/ L
    
        grad_function = self.__choose_gradient(l2)
        error_criterion = lambda X, y, w, w_prev: np.linalg.norm(grad_function(X, y, w), 2)
        #error_criterion = lambda X, y, w, w_prev: np.linalg.norm(w - w_prev, 2)
        to_seconds = lambda s: t.microseconds * 1e-6 + t.seconds
        opt = self.choose_opt_method(heavy_ball, nesterov_moment)



        self._iter       = iter
        self._step       = step
        self._beta       = beta
        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._l2_coef    = l2_coef
        self._time       = []

        if (dynamic_beta == 1):
            self._f_beta = self.beta_1
        elif (dynamic_beta == 2):
            self._f_beta = self.beta_2
        elif (dynamic_beta == 3):
            self._f_beta = self.beta_3
        else:
            self._f_beta = lambda k: self._beta

        w_prev = w0
        #time.microseconds * 1e-6 + time.seconds
        time_start = dt.now()
        for i in range(int(iter)):
            self._i = i
            self._w = opt(self.__function, grad_function, self._w,
                                            lr_func, step, error_criterion, 
                                            X_train, y, eps)
            error = error_criterion(X_train, y, self._w, w_prev)
            
            w_prev = self._w
            self._time.append(dt.now() - time_start)
            self._errors.append(error) 
            self._accuracies.append(accuracy_score(y, self.predict(X)))
            
            #print("%d: %f" %(i, error))
            
            if (error < eps):
                break
        
        return self

    def fit_test(self, X, y, X_1, y_1, eps = 5*10e-3, iter = 10 ** (6-3), l2 = False, step=1000, 
        l2_coef = 1, heavy_ball = False, nesterov_moment = False, beta = 0.1, dynamic_beta=0):

        # Добавляем ещё столбец для константы
        X_train, w0 = self.__add_constant_column(X)
        X_test, _ = self.__add_constant_column(X_1)
        y_test = y_1
        
        # Находим константу липшица для подбора learning rate
        hessian = np.zeros((X_train.shape[1], X_train.shape[1]))
        
        for x in X_train[:]:
            hessian = hessian + 1/(4*X_train.shape[0])  * np.outer(x, x)
        L = np.linalg.norm(hessian, 2)

        lr_func = lambda X, y, w: 1.0/ L
    
        grad_function = self.__choose_gradient(l2)
        to_seconds = lambda t: t.microseconds * 1e-6 + t.seconds
        error_criterion = lambda X, y, w, w_prev: np.linalg.norm(grad_function(X, y, w), 2)
        #error_criterion = lambda X, y, w, w_prev: np.linalg.norm(w - w_prev, 2)
        
        opt = self.choose_opt_method(heavy_ball, nesterov_moment)
        self._iter       = iter
        self._step       = step
        self._beta       = beta
        self._errors     = []
        self._accuracies = []
        self._w          = w0
        self._l2_coef    = l2_coef
        self._time       = []
        
        if (dynamic_beta == 1):
            self._f_beta = self.beta_1
        elif (dynamic_beta == 2):
            self._f_beta = self.beta_2
        elif (dynamic_beta == 3):
            self._f_beta = self.beta_3
        else:
            self._f_beta = lambda k: self._beta
        
        w_prev = w0
        time_start = dt.now()
        for i in range(int(iter)):
            self._i = i
            #print(opt)-#
            self._w = opt(self.__function, grad_function, self._w,
                                            lr_func, step, error_criterion, 
                                            X_train, y, eps)

            
            error = error_criterion(X_train, y, self._w, w_prev)
            
            w_prev = self._w
            self._time.append(to_seconds(dt.now() - time_start))
            self._errors.append(error) 
            self._accuracies.append(accuracy_score(y, self.predict(X)))
            
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
        y_pred = self.prob(X)
        
        for i in range(len(y_pred)):
            if (y_pred[i] >= 0.5):
                y_pred[i] = 1
            else:
                y_pred[i] = -1

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
        
        y_prob = 1/(1 + np.exp(-self._w @ X_train.T))
    
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

    def __nesterov_moment(self, f, grad_f, w0, 
                         lr, iter, error_criterion, 
                         X, y, eps):
        """
        Это метод Нестерова.
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
        - eps               : величина ошибки

        Returns:
        Наилучшую минимальную точку, которую удалось найти.
        """
    
        w = w0
        prev_w = w
        v = grad_f(X, y, w)
        
        for k in range(iter):
            beta = self._f_beta(self._i * iter + k)
            v = beta * v + (1 - beta) * grad_f(X, y, w - lr(X, y, w) * beta * v)
        
            w = w - lr(X, y, w) * v
            
            if (k % 30 == 0):
                error = error_criterion(X, y, w, prev_w)
                
                if (error < eps):
                    return w
            
            prev_w = w
            
        return w
    

    def __heavy_ball(self, f, grad_f, w0, 
                         lr, iter, error_criterion, 
                         X, y, eps):
        """
        Это метод тяжелого шарика.
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
        - eps               : величина ошибки

        Returns:
        Наилучшую минимальную точку, которую удалось найти.
        """
    
        w = w0
        prev_w = w
        v = grad_f(X, y, w)
        for k in range(iter):
            v = self._beta * v + (1 - self._beta) * grad_f(X, y, w)
            
            w = w - lr(X, y, w) * v
            
            if (k % 30 == 0):
                error = error_criterion(X, y, w, prev_w)
            prev_w = w
            if (error < eps):
                return w
        return w
    
    def __function(self, X, Y, w):
        """
        Целевая фукнция, которую минимизирует наша логистическая регрессия.

        Input: 
        - X   : матрица фичей
        - y   : вектор ответов
        - w   : параметры модели
        """
        sum = 0
        n = X.shape[0]

        for i in range(len(Y)):     
            sum = sum + 1/n * np.log(1 - np.exp(-Y[i] * X[i, :] @ w)) 
        
        return sum

    def __grad_function(self, X, Y, w):
        sum = np.zeros(w.shape)
        n = X.shape[0]
        for i in range(len(Y)):            
            sum = sum  - 1/n * Y[i] * X[i] /(1 + np.exp(Y[i] * w * X[i])) * np.exp(Y[i] * w * X[i])

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
            grad_function = lambda X, Y, w: self.__grad_function(X, Y, w) + 2 * self._l2_coef * w
        else:
            grad_function = lambda X, Y, w: self.__grad_function(X, Y, w)
        return grad_function
    
    def get_errors(self):
        return self._errors

    def get_accuracy(self):
        return self._accuracies

    def get_weights(self):
        return self._w

    def get_time(self):
        return self._time

    def choose_opt_method(self, heavy, nesterov):
        if heavy == True:
            return self.__heavy_ball
        if nesterov == True:
            return self.__nesterov_moment

        return self.__gradient_descent
    
    def beta_1(self, k):
        return k/(k+1)

    def beta_2(self, k):
        return k/(k+2)

    def beta_3(self, k):
        return k/(k+3)    
    
class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
    
    def __function(self, X, y, w, n):
        return 1/n * la.norm(w @ X.T - y, 2) ** 2

    def __grad_function(self, X, y, w, n):
        return 1/n * 2 * X.T @ (w @ X.T - y).T 
        
    def fit(self, X, y, eps = 5*10e-3, iter = 10 ** (6-2), l2 = False, step=1000):
        """
        Функция подбора параметров линейной модели для квадратичной функции потерь.

        Input: 
        - X     : матрица объектов
        - y     : вектор ответов

        Returns: none.
        """

        n, k = X.shape
        wo = np.array([])
        X_train = X

        # Добавляем ещё столбец для константы
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            w0 = np.random.rand(k+1)
        else:
            w0 = np.random.rand(k)
        
        hessian = 2 / n * X_train.T @ X_train
        wb, vb = np.linalg.eigh(hessian)
        
        lr_func = lambda X, y, w, n: 1/wb[-1]
        function = lambda w: 1/n * la.norm(w @ X_train.T - y, 2) ** 2
        A = X_train.T @ X_train
        b = X_train.T @ y
        grad_function = lambda w: 2 / n * (A @ w.T - b)
        error_criterion = lambda w: np.linalg.norm(grad_function(w), 2)

        self._errors = []
        self._accuracies = []
        self._w = w0

        for it in range(int(iter / step)):
            self._w = self.__gradient_descent(function, grad_function, self._w,
                                            lr_func,  step, error_criterion, X_train, y)
            
            error = error_criterion(self._w)
            self._errors.append(error)
            
            self._accuracies.append(accuracy_score(y, self.predict(X)))

            if (error < eps):
                break
                        
        return

    def get_errors(self):
        return self._errors

    def get_accuracies(self):
        return self._accuracies

    def predict(self, X):
        """
        Функция предсказания по признакам, уже натренированной модели.
        Input: 
        - X : объекты, по которым будем предксазывать

        Returns: предсказания на основе весов, ранее обученной линейной модели.
        """
        n, k = X.shape
        X_train = X
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        y_pred = self._w @ X_train.T
    
        for i in range(len(y_pred)):
            if (y_pred[i] >= 0.5):
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        
        return y_pred

    def get_weights(self):
        """
        Функция получения весов нашей линейной модели.

        Input: None.
        Returns: Параметры модели.
        """
        return self._w
    
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

    def __gradient_descent(self, f, grad_f, x0, 
                         lr, iter, error_criterion, 
                         X, y):
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

        Returns:
        Наилучшую минимальную точку, которую удалось найти.
        """
    
        w = x0
        eps = 5*10e-5
        n, k = X.shape 
        for k in range(iter):
            prev_w = w
            w = w - lr(X, y, w, n) * grad_f(w)
            
            error = error_criterion(w)

            if (error < eps):
                return w
        return w

    def __gradient_descent_proj(self, f, grad_f, x0, 
                         lr, iter, error_criterion, 
                         X, y):
        """
        Это модифицированный градиентный спуск, с проекций градиента на l1-шар
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

        Returns:
        Наилучшую минимальную точку, которую удалось найти.
        """
    
        w = x0
        eps = 5*10e-5
        n, k = X.shape 
        for k in range(iter):
            prev_w = w
            w = w - lr(X, y, w, n) * self.proj(grad_f(X, y, w, n))
            
            error = error_criterion(X, y, w, n)

            if (error < eps):
                return w
        return w

    def get_errors(self):
        """
        Функция получения ошибок, подсчитываемой при вызове fit.

        Input: None
        Return: лист ошибок
        """
        return self._errors

    def get_accuracy(self):
        """
        Функция получения метрики accuracy_score, подсчитываемой при вызове fit.
        
        Input: None
        Return: лист accuracy_score.
        """
        return self._accuracies
    

from datetime import datetime as dt
from sklearn.metrics import accuracy_score
class Optimizer:
    def __init__(self, func, grad_func, w0, learning_rate, iter, args, name, label):
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
        self.args['w_prev'] = self.w

    def gd(self, w, k):
        lr = self.learning_rate(w, self.func, self.grad_func, self.args)
        return w - lr * self.grad_func(w, self.args) 
    
    def heavy_ball(self, w, v, k):
        v = self.args['beta']*v + (1 - self.args['beta'])*self.grad_func(w, self.args)
        lr = self.learning_rate(w, self.func, self.grad_func, self.args)
        return w - lr*v, v
    
    def nesterov(self, w, v, k):
        lr = self.learning_rate(w, self.func, self.grad_func, self.args)
        beta = self.args['beta']
        v = beta*v + (1-beta)*self.grad_func(w - lr*beta*v, self.args)
        return w - lr*v, v

    def predict(self, X):
        return np.sign(X @ self.w)

    def fit(self):
        to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
        time_start = dt.now()
        for k in range(self.iter):
            w_prew = self.w
            if self.name == 'gd':
                self.w = self.gd(self.w, k)
            elif self.name == 'gdhv':
                self.w, self.v = self.heavy_ball(self.w, self.v, k)
            elif self.name == 'gdnesterov':
                self.w, self.v = self.nesterov(self.w, self.v, k)

            self.args['w_prev'] = w_prew
            error = np.linalg.norm(self.grad_func(self.w, self.args), 2)
            self.time.append(to_seconds(dt.now() - time_start))
            self.errors.append(error)
            self.accuracy.append(accuracy_score(self.predict(self.args['X_test']), self.args['y_test']))
            
            if error < 1e-8:
                break

