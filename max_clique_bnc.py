#!/usr/bin/env python
# coding: utf-8

import numpy as np
from docplex.mp.model import Model
import copy
import random

eps = 1e-5

class Graph(object):
    """
    class of graph
    """

    def __init__(self):
        """
        V- множество вершин
        D - множество вершин, с возможностью их сортировки по степени
        colors - нужен только для покраски по вершинам
        """
        self.V = {}
        self.D = []
        self._D_sorted = False
        self.used_colors = set()
        self.coloring = dict()

    def add_edge(self, v1, v2):
        """
        add pair of vertexes to V
        add each other to neigbours
        """
        self._add_vertex_to_neighbours(v1, v2)
        self._add_vertex_to_neighbours(v2, v1)

        return

    def add_vertex(self, v1):
        """
        add one vertex to V
        """

        if v1 in self.V:  # если вершина есть в множестве, то все ок. Ее не надо добавлять
            # МБ надо обновить ее степень?
            #             v1.update_degree = len(self.neighbours(v1))
            #             self.V[v1]['degree'] = self.degree(v1)
            #             D[self.V[v1]['D_index'] ] = self.V[v1]['degree']
            print('vertex already in graph.Vertexes')

            return
        else:
            self.V[v1] = {}
            self.V[v1]['name'] = v1
            self.V[v1]['neighbours'] = set()
            self.V[v1]['degree'] = 0
            self.D.append(self.V[v1])
            self._D_sorted = False

            return

    def _add_vertex_to_neighbours(self, v1, v2):
        """
        add vertex v2 to neigbours of vertex v1
        """
        if v1 not in self.V:
            self.add_vertex(v1)
        if v2 in self.V[v1]['neighbours']:
            print('vertex already in neighbours')
            return
        self.V[v1]['neighbours'].add(v2)
        self.V[v1]['degree'] += 1
        self._D_sorted = False

        return

    def neighbours(self, v1):
        """
        print all neigbour names of chosed vertex
        """
        if v1 in self.V:
            return self.V[v1]['neighbours']
        else:
            print("no such vertex in graph")
            return None

    def is_neighbour(self, vertex_name, neighbour_name):
        """
        vetrex - vertex name
        neighbout - neighbour name
        True if neighbour
        False if not neighbour
        """
        if neighbour_name in self.V[vertex_name]['neighbours']:
            return True
        return False

    def sort_vertex_by_degree(self):
        """
        sort vertexes: big degree is first
        """
        # добавить маркер сортер для массива Д для того, чтобы не сортировать сортированный Д
        if self._D_sorted:
            return
        self.D.sort(key=lambda input: input['degree'])
        self._D_sorted = True
        return

    def degree(self, v1):
        """
        return degree of chosen vertex
        """
        if v1 in self.V:
            return len(self.V[v1]['neighbours'])
        else:
            return None

    def clear_coloring(self):
        self.used_colors = set()
        self.coloring = dict()

    def recolor(self):
        """
        Использовать с candidates при ветвлении
        т.к в первоначальной окраске слишком много цветов
        скорее всего стоит красить не каждое ветвление, а мб только в начале
        """
        #     print('***recolor***')
        max_color = -1
        self.coloring = dict()
        for vertex_name in self.V:
            self.coloring[vertex_name] = None
        self.used_colors = dict()
        for vertex_name in self.V:
            #             print('vertex_name',vertex_name)
            #             print('used_Colors',self.used_colors)
            avalible_colors = set(self.used_colors.keys())
            for neighbour_name in g.V[vertex_name]['neighbours']:
                #                 print('N',neighbour_name, self.coloring[neighbour_name] ,)
                avalible_colors -= {self.coloring[neighbour_name]}
                if len(avalible_colors) == 0:
                    break
            if len(avalible_colors) == 0:
                max_color += 1  # color is index in candidates_coloring
                self.used_colors[max_color] = set()
                self.used_colors[max_color].add(vertex_name)
                self.coloring[vertex_name] = max_color

            #             print('avalible_colors',avalible_colors)
            if len(avalible_colors) != 0:
                for avalible_color in avalible_colors:
                    rand_avalible_color = avalible_color
                    break
                self.used_colors[rand_avalible_color].add(vertex_name)
                self.coloring[vertex_name] = rand_avalible_color

    #             print('chosen_Color',self.coloring[vertex_name])
    #         return used_colors,candidates_coloring

    def recolor_by_degree(self, rev=False):
        """
            Использовать с candidates при ветвлении
            т.к в первоначальной окраске слишком много цветов
            скорее всего стоит красить не каждое ветвление, а мб только в начале
            """
        #     print('***recolor***')    
        self.sort_vertex_by_degree()
        max_color = -1
        self.coloring = dict()
        for vertex_name in self.V:
            self.coloring[vertex_name] = None
        self.used_colors = dict()
        if rev:
            D_range = range(len(self.D) - 1, -1, -1)
        else:
            D_range = range(len(self.D))
        for vertex_index in D_range:
            vertex = self.D[vertex_index]
            vertex_name = vertex['name']
            #             print('vertex_name',vertex_name)
            #             print('used_Colors',self.used_colors)
            avalible_colors = set(self.used_colors.keys())
            for neighbour_name in g.V[vertex_name]['neighbours']:
                #                 print('N',neighbour_name, self.coloring[neighbour_name] ,)
                avalible_colors -= {self.coloring[neighbour_name]}
                if len(avalible_colors) == 0:
                    break
            if len(avalible_colors) == 0:
                max_color += 1  # color is index in candidates_coloring
                self.used_colors[max_color] = set()
                self.used_colors[max_color].add(vertex_name)
                self.coloring[vertex_name] = max_color

            #             print('avalible_colors',avalible_colors)
            if len(avalible_colors) != 0:
                for avalible_color in avalible_colors:
                    rand_avalible_color = avalible_color
                    break
                self.used_colors[rand_avalible_color].add(vertex_name)
                self.coloring[vertex_name] = rand_avalible_color

    #             print('chosen_Color',self.coloring[vertex_name])

    def recolor_best_method(self):
        self.recolor()
        c1 = len(self.used_colors)
        self.clear_coloring()

        self.recolor_by_degree(rev=False)
        c2 = len(self.used_colors)
        self.clear_coloring()

        self.recolor_by_degree(rev=True)
        c3 = len(self.used_colors)
        self.clear_coloring()

        print('g.coloring()', c1)
        print('g.coloring_by_degree()', c2)
        print('g.coloring_by_degree_reverse()', c3)

        if c1 is min(c1, c2, c3):
            self.recolor()

        if c2 is min(c1, c2, c3):
            self.recolor_by_degree(rev=False)
        if c3 is min(c1, c2, c3):
            self.recolor_by_degree(rev=True)

        print('best g.colors', len(g.used_colors))


def read_dimacs_graph(file_path):
    '''
        Parse .col file and return graph object
    '''
    g = Graph()
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c'):  # graph description
                print(*line.split()[1:])
            # first line: p name num_of_vertices num_of_edges
            elif line.startswith('p'):
                p, name, vertices_num, edges_num = line.split()
                print('{0} {1} {2}'.format(name, vertices_num, edges_num))
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                g.add_edge(int(v1), int(v2))
            else:
                continue
        return g


def candidates_recolor(candidates: set):
    """
    Использовать с candidates при ветвлении
    т.к в первоначальной окраске слишком много цветов
    скорее всего стоит красить не каждое ветвление, а мб только в начале
    """
    #     print('***candidates_recolor***')
    max_color = 0
    candidates_coloring = dict()
    for vertex_name in candidates:
        candidates_coloring[vertex_name] = 0
    used_colors = {0: set()}
    for vertex_name in candidates:
        #         print('vertex_name',vertex_name)
        #         print('used_Colors',used_colors)
        neighbour_colors = set()
        for neighbour_name in g.V[vertex_name]['neighbours'].intersection(candidates):
            #             print('N',neighbour_name, candidates_coloring[neighbour_name] , neighbour_name in candidates)
            neighbour_colors.add(candidates_coloring[neighbour_name])
            if len(neighbour_colors) == len(used_colors):
                #                 print('break')
                break
        if len(neighbour_colors) == len(used_colors):
            max_color += 1  # color is index in candidates_coloring
            used_colors[max_color] = set()
            used_colors[max_color].add(vertex_name)
            candidates_coloring[vertex_name] = max_color
        #             print(used_colors,candidates_coloring)

        #         print('avalible_colors',avalible_colors)
        elif len(neighbour_colors) < len(used_colors):
            avalible_colors = set(used_colors.keys()) - neighbour_colors
            #             print('avalible_colors',avalible_colors)
            for avalible_color in avalible_colors:
                rand_avalible_color = avalible_color
                break
            used_colors[rand_avalible_color].add(vertex_name)
            candidates_coloring[vertex_name] = rand_avalible_color
    #         print('chosen_Color',candidates_coloring[vertex_name])
    if len(candidates_coloring) != len(candidates):
        raise NameError('Not all verticies colored')
    return used_colors, candidates_coloring


def candidates_recolor_degree(candidates: set, rev=False):
    """
    Использовать с candidates при ветвлении
    т.к в первоначальной окраске слишком много цветов
    скорее всего стоит красить не каждое ветвление, а мб только в начале
    """
    #     print('***candidates_recolor
    candidates_degree_order = list(candidates)
    candidates_degree_order.sort(key=lambda input: g.V[input]['degree'], reverse=rev)
    max_color = -1
    candidates_coloring = dict()
    for vertex_name in candidates:
        candidates_coloring[vertex_name] = None
    used_colors = dict()
    for vertex_name in candidates_degree_order:
        #         print('vertex_name',vertex_name)
        #         print('used_Colors',used_colors)
        avalible_colors = set(used_colors.keys())
        for neighbour_name in g.V[vertex_name]['neighbours'].intersection(candidates):
            #             print('N',neighbour_name, candidates_coloring[neighbour_name] , neighbour_name in candidates)
            avalible_colors -= {candidates_coloring[neighbour_name]}
            if len(avalible_colors) == 0:
                break
        if len(avalible_colors) == 0:
            max_color += 1  # color is index in candidates_coloring
            used_colors[max_color] = set()
            used_colors[max_color].add(vertex_name)
            candidates_coloring[vertex_name] = max_color

        #         print('avalible_colors',avalible_colors)
        if len(avalible_colors) != 0:
            for avalible_color in avalible_colors:
                rand_avalible_color = avalible_color
                break
            used_colors[rand_avalible_color].add(vertex_name)
            candidates_coloring[vertex_name] = rand_avalible_color
    #         print('chosen_Color',candidates_coloring[vertex_name])
    return used_colors, candidates_coloring


def find_candidates(vertex_to_clique=None):
    """
    если клика пустая создает множество из всех вершин графа
    если не пустая выдает ошибку - использовать update_candidates
    """
    global clique, candidates

    if len(clique) == 0:
        candidates = set(g.V.keys())
    else:
        raise NameError('len(clique) =! 0. Use update_candidates')

    return candidates


def update_candidates(used_candidates=None, candidates=None, vertex_to_clique=None):
    """
    если клика пустая выдает ошибку - нужно использовать find_candidates
    если клика не пустая то обновляет множетсво кандидитов:
        в него включаются те вершины, которые есть в кандидатах
        и те вершины, которые есть в соседях у вершины vertex_to_clique, которая
        в данный момент добавляется в клику
    """
    global clique  # , candidates

    if len(clique) == 0:
        raise NameError('len(clique) == 0. Use find_candidates')
    if candidates is None:
        raise NameError('candidates = None')
    if used_candidates is None:
        used_candidates = set()
    else:
        if vertex_to_clique not in candidates:
            raise NameError(vertex_to_clique, 'No such vertex in candidates')
        updated_candidates = candidates.intersection(g.V[vertex_to_clique]['neighbours']) - used_candidates
    return updated_candidates


def initial_heuristic_degree():
    clique = set()
    g.sort_vertex_by_degree()
    #     print(type(clique))
    for vertex in g.D:
        #         print(type(clique),type(vertex))

        all_elements_in_clique_are_neighbours_to_vertex = True
        for element in clique:
            all_elements_in_clique_are_neighbours_to_vertex = all_elements_in_clique_are_neighbours_to_vertex and g.is_neighbour(
                vertex['name'], element)
            if not all_elements_in_clique_are_neighbours_to_vertex:
                break
        #         print(vertex['name'],all_elements_in_clique_are_neighbours_to_vertex )
        if all_elements_in_clique_are_neighbours_to_vertex:
            clique.add(vertex['name'])
    return clique


def initial_heuristic_color(rev=False):
    """
    makes clique:
    first - vericies with maximal color
    """
    clique = set()
    g.sort_vertex_by_degree()
    if rev:
        color_range = range(len(g.used_colors) - 1, -1, -1)
    else:
        color_range = range(len(g.used_colors))
    for color in color_range:
        for vertex_name in g.used_colors[color]:
            all_elements_in_clique_are_neighbours_to_vertex = True
            for element_name in clique:
                all_elements_in_clique_are_neighbours_to_vertex = all_elements_in_clique_are_neighbours_to_vertex and g.is_neighbour(
                    vertex_name, element_name)
                if not all_elements_in_clique_are_neighbours_to_vertex:
                    break
            #             print(vertex_name,all_elements_in_clique_are_neighbours_to_vertex )
            if all_elements_in_clique_are_neighbours_to_vertex:
                clique.add(vertex_name)
    return clique


def find_best_heuristic():
    init_clique = initial_heuristic_color()
    init_clique_rev = initial_heuristic_color(True)
    init_clique_degree = initial_heuristic_degree()
    len_best_heuristic = max(len(init_clique), len(init_clique_rev), len(init_clique_degree))
    return max(init_clique, init_clique_rev, init_clique_degree), len_best_heuristic


def find_best_heuristic_new():
    g.clear_coloring()
    g.recolor()
    init_clique_1 = initial_heuristic_color()
    init_clique_rev_1 = initial_heuristic_color(True)
    init_clique_degree_1 = initial_heuristic_degree()
    len_best_heuristic_1 = max(len(init_clique_1), len(init_clique_rev_1), len(init_clique_degree_1))
    best_heuristic_1 = max(init_clique_1, init_clique_rev_1, init_clique_degree_1)

    g.clear_coloring()
    g.recolor_by_degree()
    init_clique_2 = initial_heuristic_color()
    init_clique_rev_2 = initial_heuristic_color(True)
    init_clique_degree_2 = initial_heuristic_degree()
    len_best_heuristic_2 = max(len(init_clique_2), len(init_clique_rev_2), len(init_clique_degree_2))
    best_heuristic_2 = max(init_clique_2, init_clique_rev_2, init_clique_degree_2)

    g.clear_coloring()
    g.recolor_by_degree(rev=True)
    init_clique_3 = initial_heuristic_color()
    init_clique_rev_3 = initial_heuristic_color(True)
    init_clique_degree_3 = initial_heuristic_degree()
    len_best_heuristic_3 = max(len(init_clique_3), len(init_clique_rev_3), len(init_clique_degree_3))
    best_heuristic_3 = max(init_clique_3, init_clique_rev_3, init_clique_degree_3)

    len_best_heuristic = max(len_best_heuristic_1, len_best_heuristic_2, len_best_heuristic_3)
    return max(best_heuristic_1, best_heuristic_2, best_heuristic_3), len_best_heuristic


def find_normed_degree_dict(g):
    """
    for veretx with the lowes degree - 1
    for veretx with the highest degree - 0
    
    made to find vertexes with lowest degree
    for use in find_most_violated_greedy_and_degree
    
    """
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


def find_independent_sets_by_coloring(coloring_colors, vertex_coloring, candidates):
    global g
    colors = copy.deepcopy(coloring_colors)
    ind_sets_coloring = dict()
    for vertex_name in vertex_coloring:
        ind_sets_coloring[vertex_name] = {vertex_coloring[vertex_name]}
    for color in colors:
        for vertex_name in coloring_colors[color]:
            available_colors = set(colors.keys())
            available_neighbours = g.V[vertex_name]['neighbours'].intersection(candidates)
            while len(available_neighbours) > 0:
                neighbour = available_neighbours.pop()
                neighbour_color = vertex_coloring[neighbour]
                neighbour_colors_set = ind_sets_coloring[neighbour]
                available_colors -= {neighbour_color}.union(neighbour_colors_set)
                for n_color in neighbour_colors_set:
                    available_neighbours -= colors[n_color]
            check_neighbours = set()
            # есть цвета по которым может быть пересечение. Их надо проверить
            for available_color in available_colors:
                available_neighbours = g.V[vertex_name]['neighbours'].intersection(candidates)
                available_neighbours = available_neighbours.intersection(
                    colors[available_color])  # change to colors
                check_neighbours = check_neighbours.union(available_neighbours)
            for neighbour in check_neighbours:
                available_colors -= ind_sets_coloring[neighbour]
            for available_color in available_colors:
                colors[available_color].add(vertex_name)
            ind_sets_coloring[vertex_name] = ind_sets_coloring[vertex_name].union(available_colors)
    return colors, ind_sets_coloring


# CPLEX функции branch_and_cut
def ind_sets_most_violated(ind_set: set):
    """
    set is set of indexes
    """
    for vertex_index_1 in ind_set:
        for vertex_index_2 in ind_set:
            if vertex_index_1 is not vertex_index_2:
                vertex_name_1 = vertex_index_1 + 1
                vertex_name_2 = vertex_index_2 + 1
                if g.is_neighbour(vertex_name_1, vertex_name_2):
                    print(g.is_neighbour(vertex_name_1, vertex_name_2))
    print('if pusto - OK')


def find_most_violated_greedy_and_degree(random_range=4, sum_without_random=5):
    """
    Heuristical algorithm to find_most_violated_greedy_and_degree
    
    random_range - number of iterations with randomes(alpha,betta)
    sum_without_random - if summ if vertex variable more than sum_without_random,
    algorithm don`t use random iterations
    
    best_ind_set - set of vertex_indexes 
    best_sum - sum of variables for best_ind_set
    """
    global x, mdl, normed_degree_dict, x_vars
    alpha_arr = [1]
    betta_arr = [0]
    best_sum = 0
    best_ind_set = {}

    for i in range(random_range):
        alpha_arr.append(random.uniform(0, 1))
        betta_arr.append(random.uniform(0, 1))
    for alpha, betta in zip(alpha_arr, betta_arr):
        sum_x = 0
        # x_vars - array of tupples (vertex_index,vertex_variable)
        # sol.iter_var_values() - iteration by nonzero variable_values
        x_vars = [(vertex_variable.get_index(), vertex_variable) for (vertex_variable, vertex_variable_val) in
                  sol.iter_var_values()]
        x_vars.sort(key=lambda i: alpha * i[1].solution_value + betta * normed_degree_dict[i[0]],
                    reverse=True)  # sort partly by degree and by solution valuae
        ind_set = set()
        ind_set.add(x_vars[0][0])
        for vertex_index_1, vertex_variable in x_vars:  # index = vertex_name - 1
            addition_forbidden = False
            for vertex_index_2 in ind_set:
                vertex_name_1 = vertex_index_1 + 1
                vertex_name_2 = vertex_index_2 + 1
                if g.is_neighbour(vertex_name_1, vertex_name_2):
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


def build_problem():
    """
    makes new global mdl - model, x - variables of model
    by existing global graph g 
    """

    global mdl, x, g
    mdl = Model(name='max clique')
    x_range = range(len(g.V))
    # decision variables
    # x = mdl.integer_var_dict(x_range)
    x = mdl.continuous_var_dict(x_range)
    ind_sets_arr = []
    clique = set()  # for find_candidates()
    candidates = find_candidates()
    (colors, vertex_coloring) = candidates_recolor(candidates)
    ind_sets_arr.append(find_independent_sets_by_coloring(colors, vertex_coloring, candidates)[0])
    (colors, vertex_coloring) = candidates_recolor_degree(candidates)
    ind_sets_arr.append(find_independent_sets_by_coloring(colors, vertex_coloring, candidates)[0])
    (colors, vertex_coloring) = candidates_recolor_degree(candidates, rev=True)
    ind_sets_arr.append(find_independent_sets_by_coloring(colors, vertex_coloring, candidates)[0])

    for ind_sets in ind_sets_arr:
        for ind_set in ind_sets:
            # print(i, colors2[i])
            #         if len(ind_sets[ind_set]) > 1:
            mdl.add_constraint(
                mdl.sum(x[vertex_name - 1] for vertex_name in ind_sets[ind_set]) <= 1)  # vertex_index = vertex_name-1

    #     for i in range(mdl.number_of_constraints):
    #         print(mdl.get_constraint_by_index(i))

    for i in x_range:
        #         mdl.add_constraint(x[i] <= 1)
        mdl.add_constraint(x[i] >= 0)

    mdl.maximize(mdl.sum(x[i] for i in x_range))


def is_int_value(value):
    """
    True - if int
    False if float
    
    """

    if value % 1 > eps and abs(value % 1 - 1) > eps:
        return False
    return True


def is_int_solution():
    """
    True - if all variables int
    False if even one of variables is float
    """
    global sol

    sol_val = sol.get_objective_value()
    if not is_int_value(sol_val):
        return False
    for var in sol.iter_var_values():
        if not is_int_value(var[1]):
            return False
    return True


def is_a_clique():
    """
    check solution if is_int_solution() is True
    
    function check if varticies vith x = 1 are clique or not
    """
    global sol, x_vars, g

    vertex_names = [i[0].index + 1 for i in sol.iter_var_values()]  # заменить на x_vars уточнить
    for vertex_name_1 in vertex_names:
        for vertex_name_2 in vertex_names:
            if vertex_name_1 is not vertex_name_2:
                if not g.is_neighbour(vertex_name_1, vertex_name_2):
                    # print(f"{vertex_name_1,vertex_name_2} are not neighbours. Not a clique")
                    return False
    return True


def add_non_edges_constraints():
    """
    work if  is_a_clique() is False
    function add new constraints for if varticies vith x = 1 if they are not neighbours
    """
    global sol, x_vars, g
    #     non_edges = []
    vertex_names = [i[0].index + 1 for i in sol.iter_var_values()]
    for vertex_name_1 in vertex_names:
        for vertex_name_2 in vertex_names:
            if vertex_name_1 is not vertex_name_2:
                if not g.is_neighbour(vertex_name_1, vertex_name_2):
                    vertex_index_1 = vertex_name_1 - 1
                    vertex_index_2 = vertex_name_2 - 1
                    con = x[vertex_index_1] + x[vertex_index_2] <= 1
                    mdl.add_constraint(con)

                    # print(f"{vertex_name_1,vertex_name_2} are not neighbours. Added {con}")


def branch_and_cut_choose_branching_var():
    """
    x_vars - sort partly by degree and by solution valuae f
    x_vars from find_most_violated_greedy_and_degree
    algorithm choose first available float variable from sorted x_vars 
    return fraction_variable_index
    """
    global len_max_current_clique, sol, x_vars

    ind_set, x_sum = find_most_violated_greedy_and_degree(sum_without_random=0)  # to sort x_vars by var_value
    if len(ind_set) > 2:
        mdl.add_constraint(mdl.sum(x[i] for i in ind_set) <= 1)  # to sort x_vars by var_value
        sol = mdl.solve()  # to sort x_vars by var_value

    for vertex_index, vertex_var in x_vars:
        if not is_int_value(vertex_var.solution_value):
            fraction_variable_index = vertex_index
            break
    return fraction_variable_index


def branch_and_cut_search(sum_without_random=2):
    global len_max_current_clique, sol, x_vars
    sol = mdl.solve()  # нужно ли тут
    if sol is None:
        return

    if sol.get_objective_value() <= len_max_current_clique:
        return

    first_time = True  # to do first iteration on while every time

    while first_time or x_sum > 1:
        ind_set, x_sum = find_most_violated_greedy_and_degree(sum_without_random)
        if len(ind_set) <= 1:
            break
        most_violated_con = mdl.sum(x[i] for i in ind_set) <= 1
        mdl.add_constraint(most_violated_con)
        sol = mdl.solve()

        if sol is None:
            return
        if is_int_solution():
            break
        first_time = False

    if not is_int_solution():
        fraction_variable_index = branch_and_cut_choose_branching_var()
        con1 = x[fraction_variable_index] >= 1
        con2 = x[fraction_variable_index] <= 0
        mdl.add_constraint(con1)
        branch_and_cut_search(sum_without_random)
        mdl.remove_constraint(con1)

        mdl.add_constraint(con2)
        branch_and_cut_search(sum_without_random)
        mdl.remove_constraint(con2)
    else:  # solution is int
        if is_a_clique():
            current_clique = sol.get_objective_value()
            if current_clique > len_max_current_clique:
                len_max_current_clique = current_clique
                print(f"current_len_max_clique is {len_max_current_clique}")
                print(f"current_max_clique is:")
                mdl.print_solution()
        else:
            add_non_edges_constraints()
            branch_and_cut_search(sum_without_random)

if __name__ == '__main__':

    path_e_1_1 = 'test/brock200_1.clq.txt'
    path_e_1_2 = 'test/brock200_2.clq.txt'
    path_e_1_3 = 'test/brock200_3.clq.txt'
    path_e_1_4 = 'test/brock200_4.clq.txt'
    path_e_2_1 = 'test/gen200_p0.9_44.clq.txt'
    path_e_2_2 = 'test/gen200_p0.9_55.clq.txt'
    path_e_3 = 'test/hamming8-4.clq.txt'
    path_e_4 = 'test/johnson16-2-4.clq.txt'
    path_e_5 = 'test/keller4.clq.txt'
    path_e_6 = 'test/MANN_a27.clq.txt'
    path_e_7 = 'test/p_hat1000-1.clq.txt'

    # Solver
    g = read_dimacs_graph(path_e_7)
    normed_degree_dict = find_normed_degree_dict(g)
    _, len_max_current_clique = find_best_heuristic_new()
    print(f'heuristic solution:{len_max_current_clique}')
    clique = set()  # for build_problem()
    x_vars = 0
    sol = 0
    build_problem()
    sol = mdl.solve()
    branch_and_cut_search(sum_without_random=2)
