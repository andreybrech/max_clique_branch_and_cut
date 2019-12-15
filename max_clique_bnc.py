from docplex.mp.model import Model
import copy
from graph import Graph, read_dimacs_graph


def build_problem(g=Graph()):
    mdl = Model(name='max clique')
    x_range = range(len(g.V))

    # decision variables
    # x = mdl.integer_var_dict(x_range)
    x = mdl.continuous_var_dict(x_range)

    for i in g.V:
        for j in x_range:
            if (j + 1 is not i) and (j + 1 not in g.V[i]['neighbours']):
                mdl.add_constraint(x[i - 1] + x[j] <= 1)

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


class Solver:
    def __init__(self, objective, vars, init_heuristic=0):
        self.upper_bound = objective
        self.vars = vars
        self.current_best = init_heuristic

    def search(self, model):
        m = copy.deepcopy(model)
        self.branch_and_bound_search(m)
        return self.current_best, self.vars

    def branch_and_bound_search(self, model):
        s = model.solve(log_output=False)
        # print("number of constraints:", model.number_of_constraints)
        if s is None:
            print("No solution")
            return
        # log_solution(s)
        if s.get_objective_value() < self.current_best:
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
            self.branch_and_bound_search(model)
            model.remove_constraint(con1)

            # print("\nbranching:", var_dict[0], ">=", int(branching_var) + 1)
            model.add_constraint(con2)
            self.branch_and_bound_search(model)
            model.remove_constraint(con2)
        else:
            # log_solution(s)
            # print("    best known solution:", self.current_best)
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


def solve_problem():
    path_test = 'test/test2.txt'
    g = read_dimacs_graph(path_test)
    heuristic = g.find_init_heuristic()
    print("Initial heuristic:", heuristic)
    model = build_problem(g)

    sol = model.solve(log_output=True)
    if sol is not None:
        model.print_solution()

        solver = Solver(sol.get_objective_value(), sol.iter_var_values(), heuristic)
        obj, variables = solver.search(model)

        print("\n------> SOLUTION <------")
        print("Objective:", obj)
        print("Solution vars:")
        for c in variables:
            print(c[0], c[1])
        # print("\nBranch-and-Bounded from:")
        # model.print_solution()
    else:
        print("Model is infeasible")


if __name__ == '__main__':
    import timeit

    elapsed_time = timeit.timeit(solve_problem, number=1)
    print("Time in seconds: ", elapsed_time)
