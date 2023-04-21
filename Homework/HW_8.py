import numpy as np
import time

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



class Optimizer:
    def __init__(self, func, nabla_f_x, nabla_f_y, x0, y0, learning_rate, iter, args, name, criterium, z_sol=None, proj_func=None):
    
        self.func      = func
        self.nabla_f_x = nabla_f_x
        self.nabla_f_y = nabla_f_y
        self.x0        = x0
        self.y0        = y0
        self.lr        = learning_rate
        self.iter      = iter
        self.args      = args
        self.name      = name
        self.proj_func = proj_func
        self.criterium = criterium
        self.z_sol     = z_sol

    def gd(self, x_curr, y_curr, k):
        x = x_curr - self.lr(k, self.args) * self.nabla_f_x(y_curr, y_curr, self.args)
        y = y_curr + self.lr(k, self.args) * self.nabla_f_y(x_curr, y_curr, self.args)

        return x, y
    def extragrad(self, x_curr, y_curr, x_tmp, y_tmp, k):
        x_tmp_next = x_curr - self.lr(k, self.args) * self.nabla_f_x(x_tmp, y_tmp, self.args)
        y_tmp_next = y_curr + self.lr(k, self.args) * self.nabla_f_y(x_tmp, y_tmp, self.args)
        x_next = x_curr - self.lr(k, self.args)* self.nabla_f_x(x_tmp_next, y_tmp_next, self.args)
        y_next = y_curr + self.lr(k, self.args) * self.nabla_f_y(x_tmp_next, y_tmp_next, self.args)

        return x_next, y_next, x_tmp_next, y_tmp_next
    
    def prox(self, z, xi):
            ret = np.zeros(len(z))
            tmp = 0
            for z_i, xi_i in zip(z, xi):
                tmp += z_i * np.exp(-xi_i)

            for j, (z_j, xi_j) in enumerate(zip(z, xi)):
                ret[j] = (1./tmp) * z_j * np.exp(-xi_j)

            return ret
    
    def smp(self, r_x_curr, r_y_curr, k):
        w_x_curr = self.prox(r_x_curr, self.lr(k, self.args) * self.nabla_f_x(r_x_curr, r_y_curr, self.args))
        w_y_curr = self.prox(r_y_curr, -self.lr(k, self.args) * self.nabla_f_y(r_x_curr, r_y_curr, self.args))
        r_x_next = self.prox(r_x_curr, self.lr(k, self.args) * self.nabla_f_x(w_x_curr, w_y_curr, self.args))
        r_y_next = self.prox(r_y_curr, -self.lr(k, self.args) * self.nabla_f_x(w_x_curr, w_y_curr, self.args))

        return r_x_next, r_y_next, w_x_curr, w_y_curr, self.lr(k, self.args)

    def work(self):
        z_0 = np.hstack([self.x0, self.y0])
        
        x_curr = np.copy(self.x0)
        y_curr = np.copy(self.y0)
        x_tmp = np.copy(self.x0)
        y_tmp = np.copy(self.y0)
        r_x_curr = np.copy(self.x0)
        r_y_curr = np.copy(self.y0)
        self.errors = []
        self.time = []
        x_curr, y_curr = self.x0, self.y0
        start_time = time.time()
        learning_rates = []
        w_x_list, w_y_list = [], []
        
        for k in range(self.iter):
            if self.name == 'extragrad':
                x_next, y_next, x_tmp, y_tmp = self.extragrad(x_curr, y_curr, x_tmp, y_tmp, k)
            elif self.name == 'smp':
                r_x_curr, r_y_curr, w_x_curr, w_y_curr, lr = self.smp(r_x_curr, r_y_curr, k)
                learning_rates.append(lr)
                w_x_list.append(w_x_curr)
                w_y_list.append(w_y_curr)
                x_next = 1./sum(learning_rates) * sum([a * b for a, b in zip(w_x_list, learning_rates)])
                y_next = 1. / sum(learning_rates) * sum([a * b for a, b in zip(w_y_list, learning_rates)])
            else:
                x_next, y_next = self.gd(x_curr, y_curr, k)

            if not(self.proj_func is None):
                x_next = self.proj_func(x_next, self.args)
                x_tmp = self.proj_func(x_tmp, self.args)
                y_next = self.proj_func(y_next, self.args)
                y_tmp = self.proj_func(y_tmp, self.args)

            error = None
            z_next = np.hstack([x_next, y_next])
            if self.criterium == 'z_k - z^*':
                error = np.linalg.norm(z_next - self.z_sol, ord=2)
            elif self.criterium == 'Err_vi':
                error = np.max(self.args['A'] @ x_next) - np.min(self.args['A'].T @ y_next)

            self.time.append(time.time() - start_time)
            self.errors.append(error)

            x_curr = x_next
            y_curr = y_next
            
            if error <= 1e-5:
                self.x_sol = x_curr
                self.y_sol = y_curr
                break
        self.x_sol = x_curr
        self.y_sol = y_curr
    
    def solution(self):
        return self.x_sol, self.y_sol
    
    def error_time(self):
        return self.time, self.errors
    def error_iter(self):
        return np.array(range(1, len(self.errors) + 1)), self.errors
