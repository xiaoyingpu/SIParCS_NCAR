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
    # just in case
    return xi * xi + xj * xj - 2 * xi * xj + yi * yi + yj * yj - 2 * yi * yj

m = Model("MIQP")

n_item = 3
M = 10000   #large constant
x = []
y = []
r = []
NEGATIVE = -10
for i in range(n_item):
    x.append(m.addVar(lb=NEGATIVE, ub = GRB.INFINITY, obj = 1.0, name="x[{}]".format(i)))
    y.append(m.addVar(lb=NEGATIVE, ub = GRB.INFINITY, obj = 1.0, name="y[{}]".format(i)))


for tupl in itertools.combinations(range(0, 3),2):
    i, j = tupl
    print tupl
    if i < j:
        r.append(m.addVar(vtype = GRB.BINARY, obj = 1.0, name = "r[{}][{}]".format(i, j) ))

print len(r)
# z is a vector with 2 * n + n-choose-2 dimension
# for the constraints
z = x + y + r
# update variables
m.update()


# set objective
# quad expression and minimize cost function
quad_expr = QuadExpr()

for i in range(n_item):
    for j in range(i, n_item):
        d = get_distance(x[i], y[i], x[j], y[j])
        quad_expr.add(d)
m.setObjective(quad_expr,GRB.MINIMIZE)


# add constraints

w = []
h = []

for i in range(3):
    w.append(2)
    h.append(2)

C_x = [[1, -1, 0], [0, 1, -1]]
C_y = [[0, 1, -1], [-1, 0, 1]]

D_x = [[],[]]
D_y = [[],[]]


#m.optimize()
