#@title Класс логистической регрессии
import numpy as np
import scipy.linalg as la
from sklearn.metrics import accuracy_score

class MyLogisticRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
    
    def __function(self, X, Y, w, n):
        sum = 0
        for i in range(len(Y)):            
            sum = sum + 1/n * np.log(1 - np.exp(-Y[i] * X[i, :] @ w)) 
        
        return sum

    def __grad_function(self, X, Y, w, n):
        sum = np.zeros(w.shape)
        
        for i in range(len(Y)):            
            sum = sum  - 1/n * Y[i] * X[i, :] /(1 + np.exp(Y[i] * w * X[i, :]))

        return sum
        
    def __grad_function_l2(self, X, Y, w, n):
        return self.__grad_function(X, Y, w, n) + 2 * self.alpha * w

    def fit(self, X, y, eps = 5*10e-3, iter = 10 ** (6-2), l2 = False, step=1000, alpha = 1):
        """
        Функция подбора параметров линейной модели для квадратичной функции потерь.

        Input: 
        - X     : матрица объектов
        - y     : вектор ответов

        Returns: none.
        """
        
        n, k = X.shape
        X_train = X

        # Добавляем ещё столбец для константы
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            w0 = np.random.rand(k+1)
        else:
            w0 = np.random.rand(k)
        
        lr_func = lambda X, y, w, n: 5*10e-3
        grad_function = self.__grad_function

        if l2 == True:
            grad_function = self.__grad_function_l2
        
        error_criterion = lambda X, y, w, n: np.linalg.norm(grad_function(X, y, w, n), 2)
        
        self.alpha = alpha
        self._errors = []
        self._accuracies = []
        self._w = w0

        for i in range(int(iter / step)):
            self._w = self.__gradient_descent(self.__function, grad_function, self._w,
                                            lr_func, step, error_criterion, 
                                            X_train, y)
            error = error_criterion(X_train, y, self._w, n)

            self._errors.append(error) 
            
            self._accuracies.append(accuracy_score(y, self.predict(X)))
            
            if (error < eps):
                break
        
        return self
    
    def get_errors(self):
        return self._errors

    def get_accuracy(self):
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

        y_pred = 1/(1 + np.exp(-self._w @ X_train.T))
        
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

    def __gradient_descent(self, f, grad_f, w0, 
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
    
        w = w0
        eps = 5*10e-5
        n, k = X.shape 
        for k in range(iter):
            prev_w = w
            
            w = w - lr(X, y, w, n) * grad_f(X, y, w, n)
            
            if (k % 30 == 0):
                error = error_criterion(X, y, w, n)

            if (error < eps):
                return w
        return w

    def __heavy_ball(self, f, grad_f, w0, 
                         lr, iter, error_criterion, 
                         X, y, alpha):
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
    
        w = w0
        eps = 5*10e-5
        n, k = X.shape 
        for k in range(iter):
            prev_w = w
            
            w = w - lr(X, y, w, n) * grad_f(X, y, w, n)
            
            if (k % 30 == 0):
                error = error_criterion(X, y, w, n)

            if (error < eps):
                return w
        return w
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
        
        error_criterion = lambda X, y, w, n: np.linalg.norm(self.__grad_function(X, y, w, n), 2)

        self._errors = []
        self._accuracies = []
        self._w = w0

        for it in range(int(iter / step)):
            self._w = self.__gradient_descent(self.__function, self.__grad_function, self._w,
                                            lr_func,  step, error_criterion, X_train, y)
            
            error = error_criterion(X_train, y, self._w, n)
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

