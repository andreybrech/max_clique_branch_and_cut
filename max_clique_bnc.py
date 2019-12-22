from docplex.mp.model import Model
import copy
from graph import Graph, read_dimacs_graph


def build_problem(g=Graph()):
    mdl = Model(name='max clique')
    x_range = range(len(g.V))

    candidates = set(g.V.keys())
    (colors, vertex_coloring) = g.recolor(candidates)
    (colors2, ind_set_coloring) = g.find_independent_sets_by_coloring(colors, vertex_coloring, candidates)

    # decision variables
    # x = mdl.integer_var_dict(x_range)
    x = mdl.continuous_var_dict(x_range)

    for i in colors2:
        # print(i, colors2[i])
        if len(colors2[i]) > 1:
            mdl.add_constraint(mdl.sum(x[j-1] for j in colors2[i]) <= 1)

    for i in range(mdl.number_of_constraints):
        print(mdl.get_constraint_by_index(i))

    for i in x_range:
        mdl.add_constraint(x[i] <= 1)
        mdl.add_constraint(x[i] >= 0)

    mdl.maximize(mdl.sum(x[i] for i in x_range))
    return mdl


eps = 1e-5


def log_solution(res):
    print("Objective", res.get_objective_value())
    # print("solution vars:", end=' ')
    # for i, j in res.iter_var_values():
    #     print(i, j, sep=', ', end='; ')
    # print("")


def is_int_solution(sol):
    if sol.get_objective_value() % 1 != 0:
        return False
    for var in sol.iter_var_values():
        if var[1] % 1 > eps and abs(var[1] % 1 - 1) > eps:
            return False
    # print("int solution")
    # for i, j in sol.iter_var_values():
    #     print(i, j, sep=', ', end='; ')
    # print("")
    return True


def is_a_clique(sol, g=Graph()):
    var_values = [i[0].index+1 for i in sol.iter_var_values()]
    for i in range(len(var_values)):
        for j in range(i+1, len(var_values)):
            # print(vars[i], vars[j])
            # print(g.V[vars[i]]['neighbours'])
            if var_values[j] not in g.V[var_values[i]]['neighbours']:
                return False
    return True


class Solver:
    def __init__(self, objective, vars, g=Graph(),init_heuristic=0):
        self.upper_bound = objective
        self.vars = vars
        self.current_best = init_heuristic
        self.g = g

    def search(self, model):
        m = copy.deepcopy(model)
        # call solution search function
        return self.current_best, self.vars


def solve_problem():
    path_test = 'test/test2.txt'
    g = read_dimacs_graph(path_test)
    # heuristic = g.find_init_heuristic()
    # print("Initial heuristic:", heuristic)
    model = build_problem(g)

    sol = model.solve(log_output=True)
    if sol is not None:
        model.print_solution()

        solver = Solver(sol.get_objective_value(), sol.iter_var_values(), g)
        obj, variables = solver.search(model)

        print("\n------> SOLUTION <------")
        print("Objective:", obj)
        print("Solution vars:")
        for c in variables:
            print(c[0], c[1])
        # print("\nBranch-and-Cut from:")
        # model.print_solution()
    else:
        print("Model is infeasible")


if __name__ == '__main__':
    import timeit
    elapsed_time = timeit.timeit(solve_problem, number=1)
    print("Time in seconds: ", elapsed_time)
