from gurobipy import *
import numpy as np
import time
from gurobipy import *
from itertools import chain, combinations

# Add SEC for conventional formulation in a lazy way
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        node_vals = model.cbGetSolution(model._yvars)
        edge_vals = model.cbGetSolution(model._xvars)

        nodes_selected = tuplelist(i for i in node_vals if node_vals[i] > 0.5)
        edges_selected = tuplelist((i,j) for (i,j) in edge_vals if edge_vals[i,j] > 0.5)

        # find the shortest cycle in the selected edge list
        tour = subtour(nodes_selected, edges_selected, len(node_vals))

        if len(tour) < len(nodes_selected):
            expr = 0
            for i in range(len(tour)):
                for j in range(i+1, len(tour)):
                    expr += model._xvars[tour[i], tour[j]]

            expr += model._xvars[tour[len(tour)-1], tour[0]]
            # add subtour elimination constraint for every pair of cities in tour
            model.cbLazy(expr <= len(tour)-1)
            model.update();

# find shortest subtour in a list of edges
def subtour(nodes, edges, n):
    unvisited = nodes[:]
    cycle = range(n+1) # initial length has 1 more city

    while unvisited: # true if list is non-empty
        thiscycle = []
        neighbors = unvisited

        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i,j in edges.select(current,'*') if j in unvisited]

        if 0 not in thiscycle and len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

def get_static_tour(m):
    """ Obtains the tour of solved gurobi model """
    res_edges = []

    x = m._xvars
    y = m._yvars

    if m.status == GRB.Status.OPTIMAL:

        print('\nFinal: %g' % m.objVal)
        print('\nNode included:')

        for i in range(m._n):
            if y[i].x > 0.0001:
                print(i, end = " ")

        print("\nEdges used:")
        for (i,j) in m._edges:
            if x[i,j].x > 0.0001:
                print(str(i) + "-" + str(j), end =", ")
                res_edges.append((i,j))
    else:
        print('Not finished / no solution')

    return res_edges

def order_cycle_edges(edges):
    """ Orders the list of edges by their occurence in the cycle
        [(0,3), (2,0), (3,2)] -> [(0,3), (3,2), (2,0)]"""
    res = [edges[0]]
    current = edges[0]

    edges_left = set(edges)
    edges_left.remove(current)

    while len(edges_left) > 0:

        possible = [(i,j) for (i,j) in edges_left if i==current[1]]

        if len(possible) != 1:
            print("not a cycle")

        current = possible[0]
        res.append(current)
        edges_left.remove(current)

    return res

def add_SEC_F1(m):
    """ Adds first flow-based subtour elimination constraint """
    x = m._xvars
    y = m._yvars
    n = m._n
    edges = m._edges

    fl = m.addVars(edges, vtype=GRB.CONTINUOUS, name='fl_')

    # flow constraints
    for (i,j) in m._edges:
        m.addConstr(fl[i,j] <= (n-1) * x[i,j])

    m.addConstr(sum(fl[0,j] for j in range(1,n)) == (sum(y[i] for i in range(n))-1))

    for j in range(1,n):
        m.addConstr(sum(fl[i,j] for i in range(n) if i!=j) - sum(fl[j,k] for k in range(n) if j!=k) == y[j])

def add_SEC_F3(m):
    """ Adds third flow-based subtour elimination constraint """
    x = m._xvars
    y = m._yvars
    n = m._n
    edges = m._edges
    commodities = [(i,j,k) for (i,j) in edges for k in range(n)]

    fl = m.addVars(commodities, vtype=GRB.CONTINUOUS, lb=0, name='fl_')

    # commodities
    for (i,j,k) in commodities:
        m.addConstr(fl[i,j,k] <= x[i,j])

    for k in range(1,n):
        m.addConstr(sum(fl[0,i,k] for i in range(1,n)) == y[k])
        m.addConstr(sum(fl[i,0,k] for i in range(1,n)) == 0)
        m.addConstr(sum(fl[i,k,k] for i in range(n) if i!=k) == y[k])
        m.addConstr(sum(fl[k,i,k] for i in range(n) if i!=k) == 0)

        for j in range(1,n):
            if j != k:
                m.addConstr(sum(fl[i,j,k] for i in range(n) if i!=j) - sum(fl[j,i,k] for i in range(n) if i!=j) == 0)

def add_SEC_seq(m):
    """ Adds sequence-based subtour elimination constraint """
    x = m._xvars
    y = m._yvars
    n = m._n
    edges_wo_earth = [(i,j) for i in range(1,n) for j in range(1,n) if i!=j]
    u = m.addVars(range(n), vtype=GRB.INTEGER, name='u_', lb = 0, ub = n-1)
    M = n
    m.addConstrs(u[i] - u[j] + n * x[i,j] <= n-1 + M * (1 - y[i]) + M * (1 - y[j]) for (i,j) in edges_wo_earth)

def add_SEC_T1(m):
    """ Adds first time-based subtour elimination constraint """
    x = m._xvars
    y = m._yvars
    n = m._n
    edges = m._edges

    times = list(range(n))
    edge_times = [(i,j,t) for (i,j) in edges for t in times]

    z = m.addVars(edge_times, vtype=GRB.BINARY, name='z_')

     # as many edges as nodes visited
    m.addConstr(z.sum('*','*','*') == y.sum('*'))

    M = 2 * n + 1
    m.addConstrs(sum(t * z[i,j,t] for j in range(n) if j!=i for t in times) - sum(t * z[k,i,t] for k in range(n) if k!=i for t in times) + (1-y[i]) * M >= 1 for i in range(1,n))
    m.addConstrs(x[i,j] - sum(z[i,j,t] for t in times) == 0 for (i,j) in edges)

    u = m.addVars(edges, vtype=GRB.CONTINUOUS, name='fl_')

def add_SEC_T2(m):
    """ Adds second time-based subtour elimination constraint """
    x = m._xvars
    y = m._yvars
    n = m._n
    edges = m._edges

    times = list(range(n))
    edge_times = [(i,j,t) for (i,j) in edges for t in times]
    z = m.addVars(edge_times, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z_')

    # as many edges as nodes visited
    m.addConstrs(sum(z[i,j,t] for t in times for i in range(n) if i!=j) == y[j] for j in range(n))
    m.addConstrs(sum(z[i,j,t] for t in times for j in range(n) if i!=j) == y[i] for i in range(n))
    m.addConstrs(sum(z[i,j,t] for (i,j) in edges) <= 1 for t in times)

    M = 2 * n + 1
    m.addConstrs(sum(t * z[i,j,t] for j in range(n) if j!=i for t in times) - sum(t * z[k,i,t] for k in range(n) if k!=i for t in times) + (1-y[i]) * M >= 1 for i in range(1,n))

    m.addConstrs(x[i,j] - sum(z[i,j,t] for t in times) == 0 for (i,j) in edges)

def add_SEC_to_model(m, SEC_TYPE):
    if SEC_TYPE == "F1":
        add_SEC_F1(m)

    elif SEC_TYPE == "F3":
        add_SEC_F1(m)

    elif SEC_TYPE == "seq":
        add_SEC_seq(m)

    elif SEC_TYPE == "T1":
        add_SEC_T1(m)

    elif SEC_TYPE == "T2":
        add_SEC_T2(m)

    else:
        print("unsupported SEC type " + str(SEC_TYPE))


def TSP_static(A, P, budget, SEC_TYPE = "F1"):
    """ Computes a solution for a static city selection TSP with subtour elimination
        constraint SEC_TYPE """
    n = A.shape[0]
    m = Model()

    edges = [(i,j) for i in range(n) for j in range(n) if i!=j]

    # variables
    x = m.addVars(edges, vtype=GRB.BINARY, name='x_')
    y = m.addVars(range(n), vtype=GRB.BINARY, name='y_')

    m.update()

    # objective: maximize profit
    m.setObjective(sum(P[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    # incoming and outcoming edge
    m.addConstrs(x.sum(i,'*') == y[i] for i in range(n))
    m.addConstrs(x.sum('*',i) == y[i] for i in range(n))

    # edge costs static
    m.addConstr(sum(x[i,j] * A[i,j] for (i,j) in edges) <= budget)

    m._xvars = x
    m._yvars = y
    m._n = n
    m._edges = edges

    add_SEC_to_model(m, SEC_TYPE)

    # SEC
    m.update()
    m.optimize()

    return m

def TSP_static_lazy(A, P, budget):
    """ Computes a solution for a static city selection TSP with conventional
        subtour elimination with lazy addition of constraints """

    n = A.shape[0]
    m = Model()

    edges = [(i,j) for i in range(n) for j in range(n) if i!=j]

    # variables
    x = m.addVars(edges, vtype=GRB.BINARY, name='x_')
    y = m.addVars(range(n), vtype=GRB.BINARY, name='y_')

    m.update()

    # incoming and outcoming edge
    m.addConstrs(x.sum(i,'*') == y[i] for i in range(n))
    m.addConstrs(x.sum('*',i) == y[i] for i in range(n))

    # edge costs static
    m.addConstr(sum(x[i,j] * A[i,j] for (i,j) in edges) <= budget)

    # maximize profit
    m.setObjective(sum(P[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    m.Params.lazyConstraints = 1

    m._xvars = x
    m._yvars = y
    m._n = n
    m._edges = edges

    m.update()

    m.optimize(subtourelim)

    res_edges = get_static_tour(m, edges, n)

    return m

def cov_TSP_no_covering_cost(A, P, C, budget, SEC_TYPE = "F1", TIMELIMIT = 3*60*60):

    n = A.shape[0]
    m = Model()

    edges = [(i,j) for i in range(n) for j in range(n) if i!=j]

    # variables
    x = m.addVars(edges, vtype=GRB.BINARY, name='x_')
    y = m.addVars(range(n), vtype=GRB.BINARY, name='y_')
    z = m.addVars(range(n), vtype=GRB.BINARY, name='z_')

    profit = {}
    for i, yy in enumerate(y):
        profit[yy] = P[i]

    m.update()

    # incoming and outcoming edge
    m.addConstrs(x.sum(i,'*') == y[i] for i in range(n))
    m.addConstrs(x.sum('*',i) == y[i] for i in range(n))

    # edge costs static
    m.addConstr(sum(x[i,j] * A[i,j] for (i,j) in edges) <= budget)

    # maximize profit
    m.setObjective(sum(profit[i] * z[i] for i in range(n)), GRB.MAXIMIZE)

    m._xvars = x
    m._yvars = y
    m._n = n
    m._edges = edges

    add_SEC_to_model(m, SEC_TYPE)

    m.addConstrs(sum(C[i,j] * y[i] for i in range(n)) >= z[j] for j in range(n))

    m.setParam('TimeLimit', TIMELIMIT)

    # SEC
    m.update()
    m.optimize()

    return m

def cov_TSP_with_covering_cost(A, P, C, budget, SEC_TYPE = "F1", TIMELIMIT = 3*60*60):
    raise NotImplementedError("Cov-TSP with covering cost has not been implemented yet")
