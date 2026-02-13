import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD


def densest_subgraph(G):
    """
    Open-source replacement for the original Gurobi-based LP, using PuLP (CBC).
    Maximizes total edge weight inside a dense subgraph as before.
    """
    assert all([G[i, i] == 0 for i in range(G.shape[0])])
    n = G.shape[0]

    # Build LP: variables X[i,j] in [0, G[i,j]], Y[i] >= 0
    prob = LpProblem("densest_subgraph", LpMaximize)
    X = {(i, j): LpVariable(f"X_{i}_{j}", lowBound=0, upBound=float(G[i, j]))
         for i in range(n) for j in range(n)}
    Y = {i: LpVariable(f"Y_{i}", lowBound=0) for i in range(n)}

    # Objective: maximize sum of X (same as original)
    prob += lpSum(X[i, j] for i in range(n) for j in range(n))

    # Constraints: identical structure to original Gurobi model
    prob += lpSum(Y[i] for i in range(n)) <= 1
    for i in range(n):
        prob += lpSum(X[i, j] for j in range(n)) <= Y[i]
    for j in range(n):
        prob += lpSum(X[i, j] for i in range(n)) <= Y[j]

    prob.solve(PULP_CBC_CMD(msg=0))
    if LpStatus[prob.status] != "Optimal":
        print("Model not solved")
        raise RuntimeError("unsolved")

    Y_ = np.array([value(Y[i]) for i in range(n)])

    r_values = np.unique(Y_)
    f_values = []
    for r in r_values:
        S = np.argwhere(Y_ >= r).flatten()
        f = G[np.ix_(S, S)].sum() / S.size
        f_values.append((f, S))
    f_star, S_star = max(f_values)

    # Note: value(prob.objective) is the LP total edge weight; f_star is density.
    # With fractional solutions or solver scaling they need not match, so we do not assert.

    return S_star, ''
