#@title Класс линейной регрессии
import numpy as np
import scipy.linalg as la
from sklearn.metrics import accuracy_score

class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
    
    def __function(self, X, y, w, n):
        return 1/n * la.norm(w @ X.T - y, 2) ** 2

    def __grad_function(self, X, y, w, n):
        """
        Функция под
        """
        return 1/n * 2 * X.T @ (w @ X.T - y).T 
        
    def fit(self, X, y):
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
        

        iter = int(10 ** 7)
        
        hessian = 2 / n * X_train.T @ X_train
        wb, vb = np.linalg.eigh(hessian)
        
        lr_func = lambda X, y, w, n: 1/wb[-1]
        
        error_criterion = lambda X, y, w, n: np.linalg.norm(self.__grad_function(X, y, w, n), 2)
        eps = 5*10e-5
        iter = 10e6
        self._w = self.__gradient_descent(self.__function, self.__grad_function,
                                            w0, lr_func,  iter, error_criterion, 
                                            X_train, y)
        
        return self
    

    def fit_errors(self, X, y):
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
        
        error_criterion = lambda X, y, w, n: np.linalg.norm(self.__grad_function(X, y, w, n), 2)
        eps = 5*10e-5
        iter = 10e4
        errors = []
        accuracies = []
        self._w = w0

        for it in range(int(iter // 1000)):
            self._w = self.__gradient_descent(self.__function, self.__grad_function, self._w,
                                            lr_func,  1000, error_criterion, X_train, y)
            
            error = error_criterion(X_train, y, self._w, 1000)
            #print("№ ", it * 1000, "; error = ", error)
            errors.append(error)

            accuracies.append(accuracy_score(y, np.round(self.predict(X))))
            if (error < eps):
                break
                        
        return errors, accuracies

    def fit_proj(self, X, y, radius):
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
        self.radius = radius

        # Добавляем ещё столбец для константы
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            w0 = np.random.rand(k+1)
        else:
            w0 = np.random.rand(k)
        
        hessian = 2 / n * X_train.T @ X_train
        wb, vb = np.linalg.eigh(hessian)
        
        lr_func = lambda X, y, w, n: 1/wb[-1]
        
        error_criterion = lambda X, y, w, n: np.linalg.norm(self.__grad_function(X, y, w, n), 2)
        eps = 5*10e-5
        iter = 10e4
        errors = []
        accuracies = []
        self._w = w0

        for it in range(int(iter // 1000)):
            self._w = self.__gradient_descent_proj(self.__function, self.__grad_function, self._w,
                                            lr_func,  1000, error_criterion, X_train, y)
            
            error = error_criterion(X_train, y, self._w, 1000)
            #print("№ ", it * 1000, "; error = ", error)
            errors.append(error)

            accuracies.append(accuracy_score(y, np.round(self.predict(X))))
            if (error < eps):
                break
                        
        return errors, accuracies

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
        #
        #y[np.abs(x) < lambd] = x[np.abs(x) < lambd] * 0
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
            w = w - lr(X, y, w, n) * grad_f(X, y, w, n)
            
            error = error_criterion(X, y, w, n)

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

