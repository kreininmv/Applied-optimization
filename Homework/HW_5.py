#@title Класс логистической регрессии
import numpy as np
import scipy.linalg as la
from sklearn.metrics import accuracy_score
from datetime import datetime as dt
from tqdm import tqdm

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
        self._step         = step
        self._eps          = eps
        self._opt          = self.choose_opt_method(stochastic, batch, stochastic_vector)
        self._name         = name
        self._batch_size   = batch_size
        self._decreasing_lr = decreasing_lr
        self._sigma        = sigma

        return
 
    def _optimizer(self):
        to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
        time_start = dt.now()
        

        for i in range(int(self._step)):
            self._i = 1
            self._w = self._opt(self._function, self._grad_function, self._w, self._lr_func)
            
            error = self._error_criterion(self._w)
            self._time.append(to_seconds(dt.now() - time_start))
            self._errors.append(error) 
            
            #print("%d: %f" %(i, error))
            
            if (error < self._eps):
                return
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

        self._optimizer()
        
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
        for k in range(self._step):
            self._i += 1
            vec = np.ones_like(w)
            w = w - lr(w, self._i) * (grad_f(w) + self._sigma * np.random.randn(1) * vec)
            
            if (k % 30 == 0):
                error = self._error_criterion(w)
                if (error < self._eps):
                    return w
            
        return w

    def __stochastic_gradient_descent_vector(self, f, grad_f, w0, lr):    
        w = w0
        for k in range(self._step):
            self._i += 1
            w = w - lr(w, self._i) * (grad_f(w) + self._sigma * np.random.randn(w.shape[0]))
            
            if (k % 30 == 0):
                error = self._error_criterion(w)
                if (error < self._eps):
                    return w
            
        return w

    def __stochastic_gradient_descent_batch(self, f, grad_f, w0, lr):
        w = w0
        for k in range(self._iter):
            self._i += 1
            xi = self._sigma / self._batch_size * np.random.randn(self._batch_size).sum()
    
            w = w - lr(w, self._i) * (grad_f(w) + xi * np.ones_like(w))
            
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
        for k in range(self._step):
            self._i += 1
            w = w - lr(w, self._i) * grad_f(w)
            
            if (k % 30 == 0):
                error = self._error_criterion(w)
                if (error < self._eps):
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
    
    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time
    
    def get_step(self): return self._step
    
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
    def __init__(self, fit_intercept=False,  iter = 10, l2 = False, step=50, 
        l2_coef = 1, name='default', eps = 0.25, batch_size = 200, sigma=10, method = "GD", test=False):
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
        self._sigma         = sigma
        return

    def fit_test(self, X_train, y_train, X_test, y_test):

        X_train, w0 = self.__add_constant_column(X_train)
        X_test, _ = self.__add_constant_column(X_test)
        
        # Находим константу липшица для подбора learning rate
        hessian = np.zeros((X_train.shape[1], X_train.shape[1]))
        
        for x in X_train[:]:
            hessian = hessian + 1/(4*X_train.shape[0])  * np.outer(x, x)
        L = self._batch_size * np.linalg.norm(hessian, 2)
        
        
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
        
        loop = tqdm(range(int(self._iter)), total=int(self._iter), leave=False)

        for k in loop:
            for j in range(num_batches - 1):
                X_batch = X_train[j*self._batch_size: (j + 1)*self._batch_size, :]
                y_batch = y_train[j*self._batch_size: (j + 1)*self._batch_size]
            
                self._w = opt(self.__function, grad_function, self._w,
                              lr_func, error_criterion, X_batch, y_batch)
                
                error = error_criterion(X_batch, y_batch, self._w)
    
                self._time.append(to_seconds(dt.now() - time_start))
                self._errors.append(error) 
                self._accuracies.append(accuracy_score(y_test, self.predict(X_test)))
                
                if (self._test):
                    print(f"{j + k*num_batches}: error({error}), accuracy_test({self._accuracies[-1]}), accuracy_batch({accuracy_score(y_batch, self.predict(X_batch))}), accuracy_train({accuracy_score(y_train, self.predict(X_train))})")

                loop.set_description(f"Method({self._name}) Epoch[{k}/{self._iter}], [error=%.4f, acc test=%3.1f, acc batch =%3.1f, acc train=%3.1f]" % (error, 100.*self._accuracies[-1], 100.*accuracy_score(y_batch, self.predict(X_batch)), 100.*accuracy_score(y_train, self.predict(X_train))))
                
        return

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
    def __SAGA(self, f, grad_f, w0, 
              lr, error_criterion, X, y):
        """
        Метод SAGA.
        """
        
        w = w0
        #//Создаем лист из градиентов
        gradients = [-1/(X.shape[0]) * y[i] * X[i] /(1 + np.exp(-y[i] * w * X[i])) for i in range(len(y))]
        gradient = [sum(i) for i in zip(*gradients)]

        for k in range(self._step):
            
            for j in range(0, X.shape[0]):
                i = np.random.randint(X.shape[0])
                new_grad_i = -1/X.shape[0] * y[i] * X[i] /(1 + np.exp(-y[i] * w * X[i]))
                gradient = gradient - gradients[i] + new_grad_i + self._l2_coef * w
                gradients[i]  = new_grad_i
                w = w - lr(w) * gradient
            
            if (k % 30 == 0):
                error = error_criterion(X, y, w)
                if (error < self._eps):
                    return w
        return w
    
    def __SVRG(self, f, grad_f, w0, 
               lr, error_criterion, X, y):
        """
        Это метод оптимизации SVRG.
        """

        w = w0
        phi = w0
        
        for k in range(self._step):
            all_w = [w]
            grad = grad_f(X, y, w)
            
            for j in range(0, X.shape[0]):
                i = np.random.randint(X.shape[0])
                new_grad_i = -1/X.shape[0]*y[i]*X[i]*np.exp(-y[i] * w * X[i])/(1 + np.exp(-y[i] * w * X[i]))
                grad_i = -1/X.shape[0]*y[i]*X[i]*np.exp(-y[i] * w * X[i])/(1 + np.exp(-y[i] * phi * X[i]))
                
                gradient = new_grad_i - grad_i + grad + self._l2_coef * w
                
                w = w - lr(w) * gradient
                all_w.append(w)
            
            phi = np.array([sum(i)*1./len(all_w) for i in zip(*all_w)])
            all_w.clear()

            if (k % 30 == 0):
                error = error_criterion(X, y, w)
                if (error < self._eps):
                    return w
        return w
    
    def __SARAH(self, f, grad_f, w0, 
                lr, error_criterion, X, y):
        """
        Метод SARAH.
        """

        w = w0
        
        for k in range(self._step):
            all_w = [w]
            prew_w = all_w[-1] 
            v_t = grad_f(X, y, all_w[0])
            all_w.append(prew_w - lr(w) * v_t)
            
            for j in range(0, X.shape[0]):
                i = np.random.randint(X.shape[0])
                v_it = -1/X.shape[0] * y[i] * X[i] * np.exp(-y[i] * all_w[-1] * X[i]) \
                /( 1 + np.exp(-y[i] * all_w[-1] * X[i]))
                v_it_prev = -1/X.shape[0] * y[i] * X[i] * np.exp(-y[i] * prew_w * X[i]) \
                / (1 + np.exp(-y[i] * prew_w * X[i]))
                
                v_t = v_t + v_it - v_it_prev + self._l2_coef*all_w[-1]
                
                prew_w =all_w[-1] 
                all_w.append(prew_w - lr(w) * v_t)
                
            
            w  = all_w[np.random.randint(0, len(all_w))]
            all_w.clear()
            
            if (k % 30 == 0):
                error = error_criterion(X, y, w)
                if (error < self._eps):
                    return w
        return w
    
    def __stochastic_gradient_descent(self, f, grad_f, w0, lr, error_criterion, X, y):
        """
        Стохастичский градиентный спуск.
        """
    
        w = w0
        for k in range(self._step):

            vec = np.ones_like(w)
            xi = self._sigma / self._batch_size * np.random.randn(self._batch_size).sum()
            w = w - lr(w) * (grad_f(X, y, w) +  xi* vec)
            
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
    

from datetime import datetime as dt
from sklearn.metrics import accuracy_score
class Optimizer:
    def __init__(self, func, grad_func, grad_part_func, w0, learning_rate, iter, args, name, label):
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
        self.n = args['X_train'].shape[0] // args['batch_size']
        self.phi = np.array([self.w] * self.n)
        self.grad_part_func = grad_part_func
        self.g = self.grad_func(self.w, self.args)
        self.x = np.copy(self.w)
        self.args['w_prev'] = self.w
        self.prob = 1e-3

    def gd(self, w, k):
        lr = self.learning_rate(w, self.func, self.grad_func, self.args)
        return w - lr * self.grad_func(w, self.args) 
    
    def sgd(self, w, k):
        lr = self.learning_rate(w, self.func, self.grad_func, self.args)

        np.random.seed(k)
        j = np.random.randint(self.n)
        return  w - lr * self.grad_part_func(w, j, self.args)
    
    def saga(self, w, phi, k):
        lr = self.learning_rate(w, self.func, self.grad_func, self.args)

        np.random.seed(k)
        j = np.random.randint(self.n)
        phi_next = phi
        phi_next[j] = w
        g_k = self.grad_part_func(phi_next[j], j, self.args) - \
              self.grad_part_func(phi[j], j, self.args)
        sum = 0
        for i in range(self.n):
            sum += self.grad_part_func(phi[i], i, self.args)
        g_k += 1./self.n * sum

        return w - lr*g_k, phi_next

    def svrg(self, x_curr, w_curr, g_curr, k):
        lr = self.learning_rate(x_curr, self.func, self.grad_func, self.args)

        np.random.seed(k)
        j = np.random.randint(self.n)

        g_k = self.grad_part_func(x_curr, j, self.args) - self.grad_part_func(w_curr, j, self.args) + g_curr

        x_next = x_curr - lr * g_k

        prob = np.random.random()
        if self.prob >= prob:
            w_next = x_next
            g_next = self.grad_func(x_next, self.args)
        else:
            w_next = w_curr
            g_next = g_curr

        return x_next, w_next, g_next
    
    def sarah(self, x_curr, x_before, g_curr, k):
        lr = self.learning_rate(x_curr, self.func, self.grad_func, self.args)

        np.random.seed(k)
        j = np.random.randint(self.n)

        prob = np.random.random()
        if self.prob >= prob:
            g_next = self.grad_part_func(x_curr, j, self.args) \
                     - self.grad_part_func(x_before, j, self.args) + g_curr
        else:
            g_next = self.grad_func(x_curr, self.args)

        x_next = x_curr - lr * g_next

        return x_next, g_next

    def predict(self, X):
        return np.sign(X @ self.w)

    def fit(self):
        to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
        time_start = dt.now()
        for k in range(self.iter):
            w_prew = self.w
            if self.name == 'gd':
                self.w = self.gd(self.w, k)
            elif self.name == 'sgd':
                self.w = self.sgd(self.w, k)
            elif self.name == 'saga':
                self.w, self.phi = self.saga(self.w, self.phi, k)
            elif self.name == 'svrg':
                self.w, self.x, self.g = self.svrg(self.w, self.x, self.g, k)
            elif self.name == 'sarah':
                self.w, self.g = self.sarah(self.w, self.args['w_prev'], self.g, k)



            self.args['w_prev'] = w_prew
            error = np.linalg.norm(self.grad_func(self.w, self.args), 2)
            self.time.append(to_seconds(dt.now() - time_start))
            self.errors.append(error)
            self.accuracy.append(accuracy_score(self.predict(self.args['X_test']), self.args['y_test']))
            
            if error < 1e-8:
                break

