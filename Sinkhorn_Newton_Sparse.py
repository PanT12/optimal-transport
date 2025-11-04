from Alg.base import *

class SNSSolver(optimal_transport):
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


    def Hessian_compute(self):
        row_sum = np.sum(self.X, axis=1)
        column_sum = np.sum(self.X, axis=0)
        hessian = np.block([[np.diag(row_sum), self.X], [self.X.T, np.diag(column_sum)]]) / self.eta
        # print("hessian is ", hessian)
        # print("row sum is", row_sum)
        # print("column sum is ", column_sum)
        return hessian

    def Newton_stage(self, max_iter, rho):
        # self.f = self.eta * np.log(self.f)
        # self.g = self.eta * np.log(self.g)
        for i in range(max_iter):
            hessian = self.Hessian_compute()
            hessian = hessian + 1 * np.diag(np.ones(self.m+self.n)) # he also need to add this
            hessian[hessian <= rho] = 0.0
            # print("hessian is ", hessian)
            print("eigenvalue is ", np.linalg.eigvals(hessian))
            vals = np.linalg.eigvals(hessian)
            if np.any(vals < 0):
                print("val", vals)
            gradient_f = np.sum(self.X, axis=1, keepdims=False) - self.a
            gradient_g = np.sum(self.X, axis=0, keepdims=True)[0] - self.b
            gradient = np.block([gradient_f, gradient_g])
            self.grad_norm.append(np.linalg.norm(gradient))
            print("gradient is ", gradient)
            d = congujate_gradient(hessian, -gradient)
            print("d is ", d)
            print("diff is ", np.dot(hessian, d) + gradient)
            alpha = self.line_search_Armijo(d)
            # print("alpha: ", alpha)
            self.f += alpha * d[:self.m]
            self.g += alpha * d[self.m:]
            # print("f is ", self.f)
            # print("g is ", self.g)
            f1 = self.f[:, np.newaxis] * np.ones((1, self.n))
            g1 = self.g * np.ones((self.m, 1))
            self.X = np.exp((-self.C + f1 + g1) / self.eta - np.ones((self.m, self.n)))
            # self.loss.append(np.sum(self.C * self.X))
            self.loss.append(self.dual_obj(self.f, self.g))
            self.grad_norm.append(np.linalg.norm(gradient))

    def dual_obj(self, f, g):
        term1 = np.sum(self.a * f)
        term2 = np.sum(self.b * g)
        f1 = f[:, np.newaxis]
        X = np.exp((-self.C + f1 + g) / self.eta - np.ones((self.m, self.n)))
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