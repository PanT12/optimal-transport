from Alg.base import *

class Sinkhorn(optimal_transport):
    def Sinkhorn_stage(self, max_iter):
        K = np.exp(-self.C / self.eta)
        Kg = np.dot(K, self.g)
        for i in range(max_iter):
            self.X = self.f[:, np.newaxis] * K * self.g
            self.f = self.a / Kg
            self.g = self.b / (np.dot(K.T, self.f))
            Kg = np.dot(K, self.g)
            self.loss.append(np.sum(self.C * self.X))

    def Sinkhorn_log_stage(self, max_iter):
        for i in range(max_iter):
            f1 = self.f[:, np.newaxis]
            self.X = np.exp((-self.C + f1 + self.g) / self.eta - np.ones((self.m, self.n)))
            row_sum = np.sum(self.X, axis=1)
            self.f += self.eta * (np.log(self.a) - np.log(row_sum))

            f1 = self.f[:, np.newaxis]
            self.X = np.exp((-self.C + f1 + self.g) / self.eta - np.ones((self.m, self.n)))
            column_sum = np.sum(self.X, axis=0)
            self.g += self.eta * (np.log(self.b) - np.log(column_sum))
            # if np.linalg.norm(f - f_prev, ord=1) < tol and np.linalg.norm(g - g_prev, ord=1) < tol:
            #     break
            self.loss.append(np.sum(self.C * self.X))

    def dual_obj(self, f, g):
        term1 = np.sum(self.a * f)
        term2 = np.sum(self.b * g)
        f1 = f[:, np.newaxis] * np.ones((1, self.n))
        g1 = g * np.ones((self.m, 1))
        X = np.exp((-self.C + f1 + g1) / self.eta - np.ones((self.m, self.n)))
        term3 = np.sum(X) * self.eta
        return term1 + term2 - term3

    def line_search_Armijo(self, direction, sigma=0.7, gamma=0.5):
        alpha = 2.0
        iter_num = 0
        cur_obj = self.dual_obj(self.f, self.g)
        cur_grad_f = self.a - np.sum(self.X, axis=1, keepdims=False)
        cur_grad_g = self.b - np.sum(self.X, axis=0, keepdims=False)
        while iter_num <= 10:
            new_f = self.f + alpha * direction[:self.m]
            new_g = self.g + alpha * direction[self.m:]
            new_dual_obj = self.dual_obj(new_f, new_g)
            inner_prod = np.sum(cur_grad_f * direction[:self.m]) + np.sum(cur_grad_g * direction[self.m:])
            if new_dual_obj - cur_obj < gamma * alpha * inner_prod:
                alpha *= sigma
            else:
                return alpha

            iter_num += 1
        return alpha

