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
    Euclidean distance
    those x and y's are gurobi Var's.
    returns a QuadExpr
    """
    # just in case
    return xi * xi + xj * xj - 2 * xi * xj + yi * yi + yj * yj - 2 * yi * yj

def get_whij(dim_i, dim_j):
    """
    assuming dim_i < dim_j
    because of the p, q permutation ordering
    """
    return -0.5 * (dim_i + dim_j)



m = Model("MIQP")

n_item = 3
M = 10000   #large constant
LB = 0
UB = 20

x = []
y = []
r = []
for i in range(n_item):
    x.append(m.addVar(lb=LB, ub = UB, obj = 1.0, name="x[{}]".format(i)))
    y.append(m.addVar(lb=LB, ub = UB, obj = 1.0, name="y[{}]".format(i)))


for tupl in itertools.combinations(range(0, 3),2):
    i, j = tupl
    if i < j:
        r.append(m.addVar(vtype = GRB.BINARY, obj = 1.0, name = "r[{}][{}]".format(i, j) ))

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

C_x =  [[1, -1, 0], \
        [0, 1, -1]]

C_y =  [[0, 1, -1], \
        [-1, 0, 1]]

D_x =  [[1, -1, 0, -1 * M, 0, 0],\
        [1, 0, -1, 0, -1 * M, 0],\
        [0, 1, -1, 0, 0, -1 * M]]

D_y =  [[1, -1, 0, M, 0, 0],\
        [1, 0, -1, 0, M, 0],\
        [0, -1, 1, 0, 0, M]]

c = [0,0]
d_x = []
d_y = []

for i in range(3):
    # get the real hight and width <- same for all models anyways....
    wij = get_whij(1, 1)    #TODO hard-coded
    hij = get_whij(1, 1)
    d_x.append(wij)
    d_y.append(hij + M)


# for submatrix C * z <= b, constraint #5
for i in range(2):
    lhs = np.dot(C_x[i], x)
    m.addConstr(lhs, GRB.LESS_EQUAL, c[i])

    lhs = np.dot(C_y[i], y)
    m.addConstr(lhs, GRB.LESS_EQUAL, c[i])

# for submatrix D * z <= b
for i in range(3):
    lhs = np.dot(D_x[i], x + r)
    m.addConstr(lhs, GRB.LESS_EQUAL, d_x[i])
    lhs = np.dot(D_y[i], y + r)
    m.addConstr(lhs, GRB.LESS_EQUAL, d_y[i])
print "all constraints"
print m.getConstrs()

m.optimize()

for v in m.getVars():
    print("{} = {}".format(v.varname, v.x))
print(m.objVal)
