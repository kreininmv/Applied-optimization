import numpy as np


def generate_matrix(d, mu, L):
    tmp = np.random.rand(d, d)  # генерим случайную матрицу
    tmp = tmp + tmp.T           # делаем её симметричной
    u, s, vh = np.linalg.svd(tmp)      # раскладываем по svd
    s[0] = L
    s[-1] = mu
    for i in range(2, d):
        s[i] = mu + np.random.random() * (L - mu)

    D = np.diag(s)
    A = u.T @ D @ u

    return A

def estimate_L_logistic(X_train):
    A = 1 / (4 * X_train.shape[0]) * X_train.transpose() @ X_train
    w, v = np.linalg.eigh(A)
    L = max(w)
    return L

def f_logistic(w, X, y):
    res = 0
    for i in range(X.shape[0]):
        res += 1./X.shape[0] * np.log(1 + np.exp(- w.dot(X[i]) * y[i]))
    return res

def grad_f_logistic(w, X, y):
    res = np.zeros(w.size)
    for i in range(X.shape[0]):
        res += -1./ X.shape[0]* X[i] * y[i] * np.exp(- w.dot(X[i]) * y[i]) / (1 + np.exp(- w.dot(X[i]) * y[i]))
    return res 


class Client:
    def __init__(self, d, X, y, Q_operator='full'):
        self.w = np.zeros(d)
        self.d = d
        self.X = X
        self.y = y

        self.Q_operator = Q_operator
        self.mode = 'gd'
        self.func = 'quad'

    def get_full_grad(self, w):
        if self.func == 'quad':
            return self.X @ w - self.y
        elif self.func == 'logistic':
            return grad_f_logistic(w, self.X, self.y)
    
    def reset_error(self, eta):
        self.e = np.zeros(self.d)
        self.eta = eta

    def reset_DIANA(self, alpha):
        self.h = np.zeros(self.d)
        self.alpha = alpha

    def _Q(self, v):
        res = v
        if self.Q_operator == "full":
            pass
        elif self.Q_operator == "Rand1%":
            I = np.random.choice(self.d, size=int(self.d - 0.01 * self.d), replace=False)
            for i in I:
                res[i] = 0
        elif self.Q_operator == "Rand5%":
            I = np.random.choice(self.d, size=int(self.d - 0.05 * self.d), replace=False)
            for i in I:
                res[i] = 0
        elif self.Q_operator == "Rand10%":
            I = np.random.choice(self.d, size=int(self.d - 0.1 * self.d), replace=False)
            for i in I:
                res[i] = 0
        elif self.Q_operator == "Rand20%":
            I = np.random.choice(self.d, size=int(self.d - 0.2 * self.d), replace=False)
            for i in I:
                res[i] = 0
        elif self.Q_operator == "Top10%":
            top10 = np.sort(np.abs(v))[-10]
            res = np.array([(0.0 if abs(x) < top10 else x) for x in v])
        return res

    def grad(self, w):
        res = np.zeros(self.d)
        if self.mode == "error":
            g = self.get_full_grad(w)
            res = self._Q(self.e + self.eta * g)
            self.e = self.e + self.eta * g - res
        elif self.mode == "GD":
            res = self._Q(self.get_full_grad(w))
        elif self.mode == "DIANA":
            Delta = self.get_full_grad(w) - self.h
            res = self._Q(Delta)
            self.h = self.h + self.alpha * res
        return res

class Server:
    def __init__(self, d, clients, X_test=None, y_test=None, Q_operator="full"):
        self.w = np.zeros(d)
        self.d = d

        self.Q_operator = Q_operator
        self.errors_mode = False

        self.info_per_send = self.d
        self.set_info_per_send()
        self.clients = clients

        self.converge = []
        self.infos = []
        self.info = 0

        self.accuracies = []
        self.X_test = X_test
        self.y_test = y_test

        self.lamb = 0

    def set_info_per_send(self):
        if self.Q_operator == "full":
            self.info_per_send = self.d
        elif self.Q_operator == "Rand1%":
            self.info_per_send = self.d * 0.01
        elif self.Q_operator == "Rand5%":
            self.info_per_send = self.d * 0.05
        elif self.Q_operator == "Rand10%" or self.Q_operator == "Top10%":
            self.info_per_send = self.d * 0.1
        elif self.Q_operator == "Rand20%":
            self.info_per_send = self.d * 0.2

    def set_Q_operator(self, Q_operator):
        self.Q_operator = Q_operator
        for cl in self.clients:
            cl.Q_operator = Q_operator
        self.set_info_per_send()

    def set_func(self, func):
        for cl in self.clients:
            cl.func = func
    
    def reset(self):
        self.w = np.zeros(self.d)
        self.converge = []
        self.accuracies = []
        self.infos = []
        self.info = 0

    def get_accuracy(self):
        count = 0
        for i in range(self.y_test.size):
            if self.X_test[i].dot(self.w) * self.y_test[i] > 0:
                count += 1
        return count / self.y_test.size

    def GD(self, steps, gamma):
        self.reset()

        for cl in self.clients:
            cl.mode = "GD"

        n = len(self.clients)
        for _ in range(steps):
            full_grad = 1 / n * sum([cl.get_full_grad(self.w) for cl in self.clients])
            self.converge.append(np.linalg.norm(full_grad) ** 2)
            
            if not (self.X_test is None):
                self.accuracies.append(self.get_accuracy())
            
            self.infos.append(self.info)
            self.info += n * self.info_per_send

            grad = 1 / n * sum([cl.grad(self.w) for cl in self.clients])
            self.w = self.w - gamma * (grad + self.w * self.lamb)

    def errors(self, steps, gamma):
        self.reset()
        
        for cl in self.clients:
            cl.reset_error(gamma)
            cl.mode = "error"

        n = len(self.clients)
        for _ in range(steps):
            full_grad = 1 / n * sum([cl.get_full_grad(self.w) for cl in self.clients])
            self.converge.append(np.linalg.norm(full_grad) ** 2)
            if not (self.X_test is None):
                self.accuracies.append(self.get_accuracy())
            
            self.infos.append(self.info)
            self.info += n * self.info_per_send

            grad = 1 / n * sum([cl.grad(self.w) for cl in self.clients])
            self.w = self.w - gamma * (grad + self.w * self.lamb)
    
    def DIANA(self, steps, gamma, alpha):
        self.reset()
        
        for cl in self.clients:
            cl.reset_DIANA(alpha)
            cl.mode = "DIANA"

        n = len(self.clients)

        self.h = np.zeros(self.d)
        for _ in range(steps):
            full_grad = 1 / n * sum([cl.get_full_grad(self.w) for cl in self.clients])
            self.converge.append(np.linalg.norm(full_grad) ** 2)
            self.accuracies.append(self.get_accuracy())
            
            self.infos.append(self.info)
            self.info += n * self.info_per_send

            Delta = 1 / n * sum([cl.grad(self.w) for cl in self.clients])
            grad = self.h + Delta
            self.w = self.w - gamma * (grad + self.w * self.lamb)
            self.h = self.h + alpha * Delta
    
    def get_info_converge(self):
        x_plot = self.infos
        y_plot = self.converge
        return x_plot, y_plot
    def get_info_accuracies(self):
        x_plot = self.infos
        y_plot = self.accuracies
        return x_plot, y_plot