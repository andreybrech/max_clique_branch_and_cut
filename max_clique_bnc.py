from docplex.mp.model import Model
import copy
from graph import Graph, read_dimacs_graph
import random


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
    print("solution vars:", end=' ')
    for i, j in res.iter_var_values():
        print(i, j, sep=', ', end='; ')
    print("")


def is_int_solution(sol):
    sol_val = sol.get_objective_value()
    if sol_val % 1 > eps and abs(sol_val % 1 - 1) > eps:
        return False
    for var in sol.iter_var_values():
        if var[1] % 1 > eps and abs(var[1] % 1 - 1) > eps:
            return False
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


def get_non_edges(sol, g=Graph()):
    non_edges = []
    var_values = [i[0] for i in sol.iter_var_values()]
    for i in range(len(var_values)):
        for j in range(i+1, len(var_values)):
            if var_values[j].index+1 not in g.V[var_values[i].index+1]['neighbours']:
                non_edges.append((var_values[j], var_values[i]))
    return non_edges


class Solver:
    def __init__(self, objective, vars, g=Graph(),init_heuristic=0):
        self.upper_bound = objective
        self.vars = vars
        self.current_best = init_heuristic
        self.g = g

    def search(self, model):
        self.branch_and_cut_search(model)
        return self.current_best, self.vars

    def branch_and_cut_search(self, model):
        s = model.solve(log_output=False)
        if s is None:
            # print("No solution")
            return
        # log_solution(s)
        # print("    best known solution:", self.current_best)
        if s.get_objective_value() < self.current_best:
            # print("objective <= current best", s.get_objective_value(), self.current_best)
            return

        if not is_int_solution(s):
            if s.get_objective_value() > self.upper_bound:
                return
            # branching
            # log_solution(s)
            # print("    best known solution:", self.current_best)
            var_dict = tuple()
            branching_var = 0
            for dic in s.iter_var_values():
                if not var_dict:
                    var_dict = dic
                    branching_var = dic[1]
                else:
                    if dic[1] % 1 > eps and 1 - dic[1] % 1 > eps:
                        if dic[1] % 1 > branching_var % 1:
                            var_dict = dic
                            branching_var = dic[1]

            con1 = var_dict[0] <= int(branching_var)
            con2 = var_dict[0] >= (int(branching_var) + 1)
            # print("\nbranching:", var_dict[0], "<=", int(branching_var))
            model.add_constraint(con1)
            self.branch_and_cut_search(model)
            model.remove_constraint(con1)

            # print("\nbranching:", var_dict[0], ">=", int(branching_var) + 1)
            model.add_constraint(con2)
            self.branch_and_cut_search(model)
            model.remove_constraint(con2)
        else:
            if not is_a_clique(s, self.g):
                non_edges = get_non_edges(s, self.g)
                for non_edge in non_edges:
                    model.add_constraint(non_edge[0] + non_edge[1] <= 1)
                self.branch_and_cut_search(model)
            else:
                if s.get_objective_value() > self.current_best:
                    self.current_best = s.get_objective_value()
                    self.vars = s.iter_var_values()
                    print("* New best:", self.current_best)
                    for i, j in s.iter_var_values():
                        print(i, j, sep=', ', end='; ')
                    print("")
                else:
                    # print(s.get_objective_value(), "<= current_best(", self.current_best, ")")
                    pass

    def find_most_violated_greedy_and_degree(self, model, random_range=4, sum_without_random=5):
        global x, mdl, normed_degree_dict
        alpha_arr = [1]
        betta_arr = [0]
        best_sum = 0
        best_ind_set = {}

        for i in range(random_range):
            alpha_arr.append(random.uniform(0, 1))
            betta_arr.append(random.uniform(0, 1))
        for alpha, betta in zip(alpha_arr, betta_arr):
            sum_x = 0
            var = list(x.items())  # var - array of tupples (vertex_index,vertex_variable)
            var = [i for i in var if i[1].solution_value > 0]
            var.sort(key=lambda i: alpha * i[1].solution_value + betta * normed_degree_dict[i[0]], reverse=True)

            ind_set = set()
            ind_set.add(var[0][0])
            for vertex_index_1, vertex_variable in var:  # index = vertex_name - 1
                addition_forbidden = False
                for vertex_index_2 in ind_set:
                    if g.is_neighbour(vertex_index_1, vertex_index_2):
                        addition_forbidden = True
                        break
                if not addition_forbidden:
                    ind_set.add(vertex_index_1)
                    sum_x += vertex_variable.solution_value
            if sum_x > best_sum:
                best_sum = sum_x
                best_ind_set = ind_set
            if sum_x > sum_without_random:
                break
        return best_ind_set, best_sum


def solve_problem(path):
    #path_test = 'test/test2.txt'
    g = read_dimacs_graph(path)
    heuristic = g.find_init_heuristic()
    print("Initial heuristic:", heuristic)
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
    else:
        print("Model is infeasible")


def find_degree_dict(g):
    degree_arr = []
    for vertex_name in g.V:
        (vertex_name - 1, g.V[vertex_name]['degree'])
        degree_arr.append(
            (vertex_name - 1, g.V[vertex_name]['degree']))  # degree_arr - array of tupples (vertex_index,vertex_deg)
    degree_arr.sort(key=lambda i: i[1], reverse=False)

    min_degree = degree_arr[0][1]
    max_degree = degree_arr[-1][1]

    normed_degree_dict = dict()
    for index in range(len(degree_arr)):
        vertex_index = degree_arr[index][0]
        degree = degree_arr[index][1]
        normed_degree = 1 - (degree - min_degree) / (max_degree - min_degree)

        normed_degree_dict[vertex_index] = normed_degree
    return normed_degree_dict


normed_degree_dict = find_degree_dict(g)


def find_most_violated_greedy_and_degree(random_range=4, sum_without_random=5):
    global x, mdl, normed_degree_dict
    alpha_arr = [1]
    betta_arr = [0]
    best_sum = 0
    best_ind_set = {}

    for i in range(random_range):
        alpha_arr.append(random.uniform(0, 1))
        betta_arr.append(random.uniform(0, 1))
    for alpha, betta in zip(alpha_arr, betta_arr):
        sum_x = 0
        var = list(x.items())  # var - array of tupples (vertex_index,vertex_variable)
        var = [i for i in var if i[1].solution_value > 0]
        var.sort(key=lambda i: alpha * i[1].solution_value + betta * normed_degree_dict[i[0]], reverse=True)

        ind_set = set()
        ind_set.add(var[0][0])
        for vertex_index_1, vertex_variable in var:  # index = vertex_name - 1
            addition_forbidden = False
            for vertex_index_2 in ind_set:
                if g.is_neighbour(vertex_index_1, vertex_index_2):
                    addition_forbidden = True
                    break
            if not addition_forbidden:
                ind_set.add(vertex_index_1)
                sum_x += vertex_variable.solution_value
        if sum_x > best_sum:
            best_sum = sum_x
            best_ind_set = ind_set
        if sum_x > sum_without_random:
            break
    return best_ind_set, best_sum


if __name__ == '__main__':
    import timeit
    path = 'test/brock200_1.clq.txt'
    solve_problem(path)
    #elapsed_time = timeit.timeit(solve_problem, number=1)
    #print("Time in seconds: ", elapsed_time)

