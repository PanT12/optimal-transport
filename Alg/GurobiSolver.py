from gurobipy import *
import numpy as np


class Gurobisolver():
    def __init__(self, C, a, b):
        self.C = C
        self.a = a
        self.m = len(a)
        self.b = b
        self.n = len(b)

    def Gurobi_Solver_original(self):
        self.model = Model()
        self.model.Params.Method = 1 # Use the barrier method
        self.model.setParam('OutputFlag', 0)
        X = self.model.addVars(self.m, self.n, lb=0, vtype=GRB.CONTINUOUS, name="X")
        self.model.addConstrs(
            (quicksum(X[i, j] for j in range(self.n)) == self.a[i] for i in range(self.m)),
            name="a"
        )
        self.model.addConstrs(
            (quicksum(X[i, j] for i in range(self.m)) == self.b[j] for j in range(self.n)),
            name="b"
        )
        obj = quicksum(self.C[i, j] * X[i, j] for i in range(self.m) for j in range(self.n))
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.optimize()
        if self.model.status == GRB.Status.OPTIMAL:
            X_values = np.array([v.X for v in X.values()]).reshape(self.m, self.n)
            return X_values
            # print(np.sum(self.C * X_values))
            # for constr in self.model.getConstrs():
            #     print(f"Constraint {constr.constrName}: Dual Variable = {constr.pi}")
        else:
            print("something wrong")