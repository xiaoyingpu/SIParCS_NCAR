from gurobipy import *
import itertools

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
    print tupl
    i, j = tupl
    r.append(m.addVar(vType = GRB.BINARY, obj = 1.0, name = "r_{},{}".format(i, j) ))

z = x + y + r

m.update()


# set objective
# quad expression and minimize cost function
quad_expr = None

m.setObjective(quad_expr,GRB.MINIMIZE)





# add constraints
m.optimize()
