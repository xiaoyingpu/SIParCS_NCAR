from gurobipy import *
import itertools
from scipy.special import comb
import numpy as np



#def get_mtrx_L(n):
#    I_n = np.identity(n)
#    ones = np.ones(n)
#    return n * I_n - ones
#
#
#def get_mtrx_Q(n):
#    size_r = comb(n, 2, exact = True) * 2
#    L = get_mtrx_L(n)
#



def get_distance(xi, yi, xj, yj):
    """
    those x and y's are gurobi Var's.
    returns a QuadExpr
    """
    return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)




m = Model("MIQP")

n_item = 3
x = []
y = []
r = []
NEGATIVE = -10
for i in range(n_item):
    x.append(m.addVar(lb=NEGATIVE, ub = GRB.INFINITY, obj = 1.0, name="x_{}".format(i)))
    y.append(m.addVar(lb=NEGATIVE, ub = GRB.INFINITY, obj = 1.0, name="y_{}".format(i)))


for tupl in itertools.combinations(range(0, 4),2):
    i, j = tupl
    r.append(m.addVar(vtype = GRB.BINARY, obj = 1.0, name = "r_{},{}".format(i, j) ))

z = x + y + r

m.update()


# set objective
# quad expression and minimize cost function
quad_expr = QuadExpr()

for i in range(n_item):
    for j in range(i, n_item):
        #print "{}, {}".format(i,j)
        quad_expr.add()


#m.setObjective(quad_expr,GRB.MINIMIZE)





# add constraints
#m.optimize()
