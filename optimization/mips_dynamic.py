from optimization.mips_static import get_static_tour
from gurobipy import *

def get_times_for_static_extension(m):
    n = m._n
    y = m._yvars
    t = m._tvars

    res_times = {}

    if m.status == GRB.Status.OPTIMAL:

        print("\nTimes:")
        for i in range(n):
            if y[i].x > 0.0001:
                print(str(i) + ": " + str(t[i].x), end =", ")
                res_times[i] = t[i].x
    else:
        print('Not finished / no solution')

    return res_times

def get_times_DF(m):

    n = m._n
    x = m._yvars
    fl = m._flvars

    res_times = {}

    if m.status == GRB.Status.OPTIMAL:

        print("\nTimes:")
        for (i,j) in fl:
            if f[(i,j)].x > 0.0001:
                print(str(i) + ": " + str(fl[i,j].x), end =", ")
                res_times[i,j] = fl[i,j].x
    else:
        print('Not finished / no solution')

def order_cycle_edges(edges):

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

def get_times_DT1(m):

    res_edge_times = []

    all_cost = 0

    edges = m._edges
    y = m._yvars
    z = m._zvars
    n = m._n
    times = m._times

    if m.status == GRB.Status.OPTIMAL:

        print('\nFinal: %g' % m.objVal)
        print('\nNode included:')

        for i in range(n):
            if y[i].x > 0.0001:
                print(i, end = " ")

        print("\nEdges used:")

        for t in times:
            for (i,j) in edges:
                if z[i,j,t].x > 0.0001:
                    print(str(i) + "-" + str(j) + " at " + str(t))
                    res_edge_times.append((i,j,t))
    else:
        print('No solution')

    return res_edge_times

def compute_budget_dynamic(A, edge_times):
    return sum(A[i,j,int(t)] for (i,j,t) in edge_times)

def TSP_dynamic_add_to_static(A, P, budget, transfer_duration, function_params, SEC_TYPE = "F1"):

    times = list(range(A.shape[2]))

    n = A.shape[0]
    m = Model()

    edges = [(i,j) for i in range(n) for j in range(n) if i!=j]

    # variables
    x = m.addVars(edges, vtype=GRB.BINARY, name='x_')
    y = m.addVars(range(n), vtype=GRB.BINARY, name='y_')
    t = m.addVars(range(n), vtype=GRB.CONTINUOUS, name='t_', lb = 0, ub=times[-1])
    c = m.addVars(edges, vtype=GRB.CONTINUOUS, name='c_', lb = 0.0)

    m.update()

    # incoming and outcoming edge
    m.addConstrs(x.sum(i,'*') == y[i] for i in range(n))
    m.addConstrs(x.sum('*',i) == y[i] for i in range(n))

    # edge costs dynamic
    M = 2 * A.shape[2] + transfer_duration
    m.addConstrs(t[i] + transfer_duration <= t[j] + M * (1 - x[i,j]) for (i,j) in edges if j != 0)

    M = 3 * np.amax(A)
    m.addConstrs(c[i,j] >= function_params[i,j,0] + function_params[i,j,1] * t[i] - M * (1-x[i,j]) for (i,j) in edges)
    m.addConstr(c.sum() <= budget)

    # maximize profit
    m.setObjective(sum(P[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    m._xvars = x
    m._yvars = y
    m._tvars = t
    m._cvars = c
    m._n = n
    m._edges = edges

    add_SEC_to_model(m, SEC_TYPE)

    # SEC
    m.update()
    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        sol = get_static_tour(m, edges, n)
        res_times = get_times_for_static_extension(m)

        print("times of transfers: " + str(res_times))

    return m

def TSP_DT1(A, P, budget, transfer_duration):

    n = A.shape[0]
    m = Model()

    edges = [(i,j) for i in range(n) for j in range(n) if i!=j]
    times = list(range(A.shape[2]))

    edge_times = [(i,j,t) for (i,j) in edges for t in times]

    # variables
    y = m.addVars(range(n), vtype=GRB.BINARY, name='y_')
    z = m.addVars(edge_times, vtype=GRB.BINARY, name='z_')

    m.update()

    # maximize profit
    m.setObjective(sum(P[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    # incoming and outgoing
    m.addConstrs(z.sum(i, '*', '*') == y[i] for i in range(n))
    m.addConstrs(z.sum('*', i, '*') == y[i] for i in range(n))

    ### time constraints
    m.addConstrs(sum(t * z[i,j,t] for j in range(n) if j!=i for t in times) - sum(t * z[k,i,t] for k in range(n) if k!=i for t in times) >= transfer_duration * y[i] for i in range(1,n))

    # budget
    m.addConstr(sum(z[i,j,t] * A[i,j,t] for (i,j) in edges for t in times) <= budget)

    m._n = n
    m._yvars = y
    m._zvars = z
    m._edges = edges
    m._times = times

    m.update()

    m.optimize()

    return m

def TSP_DF1(A, P, budget):

    n = A.shape[0]
    m = Model()

    edges = [(i,j) for i in range(n) for j in range(n) if i!=j]

    # variables
    x = m.addVars(edges, vtype=GRB.BINARY, name='x_')
    y = m.addVars(range(n), vtype=GRB.BINARY, name='y_')
    fl = m.addVars(edges, vtype=GRB.INTEGER, name='fl_')
    c = m.addVars(edges, vtype=GRB.CONTINUOUS, name='c_', lb = 0.0)

    edge_times = [(i,j,t) for (i,j) in edges for t in range(A.shape[2])]
    fl_eq_t = m.addVars(edge_times, vtype = GRB.BINARY, name='absolute')

    m.update()

    # maximize profit
    m.setObjective(sum(P[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    # incoming and outcoming edge
    m.addConstrs(x.sum(i,'*') == y[i] for i in range(n))
    m.addConstrs(x.sum('*',i) == y[i] for i in range(n))

    # flow constraints
    for (i,j) in edges:
        m.addConstr(fl[i,j] <= A.shape[2] * x[i,j])

        if i != 0 and j != 0:
            m.addConstr(sum(fl_eq_t[i,j,t] for t in range(1,A.shape[2])) == x[i,j])

        for t in range(A.shape[2]):
            M = 2 * max(A.shape)

            m.addConstr(fl[i,j]-t <= 0 + M*(1-fl_eq_t[i,j,t]))
            m.addConstr(t-fl[i,j] <= 0 + M*(1-fl_eq_t[i,j,t]))

    M = np.amax(A)
    for (i,j) in edges:
        for t in range(1,A.shape[2]):
            m.addConstr(M * (2-x[i,j]-fl_eq_t[i,j,t]) + c[i,j] >= A[i,j,t])

    m.addConstr(c.sum() <= budget)
    m.addConstr(sum(fl[0,j] for j in range(1,n)) == (sum(y[i] for i in range(n))-1))

    for j in range(1,n):
        m.addConstr(sum(fl[i,j] for i in range(n) if i!=j) - sum(fl[j,k] for k in range(n) if j!=k) == y[j])

    m._xvars = x
    m._yvars = y
    m._flvars = fl

    m.update()
    m.optimize()
    res_edges = get_static_tour(m, edges, n)

    return m

def TSP_DF2(A, P, budget, function_params):

    n = A.shape[0]
    m = Model()

    edges = [(i,j) for i in range(n) for j in range(n) if i!=j]

    # variables
    x = m.addVars(edges, vtype=GRB.BINARY, name='x_')
    y = m.addVars(range(n), vtype=GRB.BINARY, name='y_')
    fl = m.addVars(edges, vtype=GRB.INTEGER, name='fl_')
    c = m.addVars(edges, vtype=GRB.CONTINUOUS, name='c_', lb = 0.0)

    m.update()

    # incoming and outcoming edge
    m.addConstrs(x.sum(i,'*') == y[i] for i in range(n))
    m.addConstrs(x.sum('*',i) == y[i] for i in range(n))

    for (i,j) in edges:
        m.addConstr(fl[i,j] <= A.shape[2] * x[i,j])

    # budget
    M = 3 * np.amax(A)
    m.addConstrs(c[i,j] >= function_params[i,j,0] + function_params[i,j,1] * (y.sum() - fl[i,j]) - M * (1-x[i,j]) for (i,j) in edges)
    m.addConstr(c.sum() <= budget)

    m.addConstr(sum(fl[0,j] for j in range(1,n)) == (sum(y[i] for i in range(n))-1))

    for j in range(1,n):
        m.addConstr(sum(fl[i,j] for i in range(n) if i!=j) - sum(fl[j,k] for k in range(n) if j!=k) == y[j])

    # maximize profit
    m.setObjective(sum(P[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    m._xvars = x
    m._yvars = y

    m.update()

    m.optimize()
    res_edges = get_static_tour(m, edges, n)

    return m

def TSP_dynamic(A, P, budget, transfer_duration=1, SEC_TYPE="DT1", regr_params=[]):
    if SEC_TYPE == "DT1":
        return TSP_DT1(A, P, budget, transfer_duration)

    elif SEC_TYPE == "DF1":
        return TSP_DF1(A, P, budget)

    elif SEC_TYPE == "DF2":
        return TSP_DF2(A, P, budget, regr_params)

    else:
        return TSP_dynamic_add_to_static(A, P, budget, transfer_duration, regr_params, SEC_TYPE = SEC_TYPE)
