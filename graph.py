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
        add each other to neighbours
        """
        self._add_vertex_to_neighbours(v1, v2)
        self._add_vertex_to_neighbours(v2, v1)

        return

    def add_vertex(self, v1):
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
        add vertex v2 to neighbours of vertex v1
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
        print all neighbour names of chosen vertex
        """
        if v1 in self.V:
            return self.V[v1]['neighbours']
        else:
            print("no such vertex in graph")
            return None

    def is_neighbour(self, vertex_name, neighbour_name):
        """
        vertex - vertex name
        neighbour - neighbour name
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
            available_colors = set(self.used_colors.keys())
            for neighbour_name in self.V[vertex_name]['neighbours']:
                #                 print('N',neighbour_name, self.coloring[neighbour_name] ,)
                available_colors -= {self.coloring[neighbour_name]}
                if len(available_colors) == 0:
                    break
            if len(available_colors) == 0:
                max_color += 1  # color is index in candidates_coloring
                self.used_colors[max_color] = set()
                self.used_colors[max_color].add(vertex_name)
                self.coloring[vertex_name] = max_color

            #             print('available_colors',available_colors)
            if len(available_colors) != 0:
                for available_color in available_colors:
                    rand_available_color = available_color
                    break
                self.used_colors[rand_available_color].add(vertex_name)
                self.coloring[vertex_name] = rand_available_color

    #             print('chosen_Color',self.coloring[vertex_name])
    #         return used_colors,candidates_coloring

    def recolor_by_degree(self):
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
        for vertex in self.D:
            vertex_name = vertex['name']
            #             print('vertex_name',vertex_name)
            #             print('used_Colors',self.used_colors)
            available_colors = set(self.used_colors.keys())
            for neighbour_name in self.V[vertex_name]['neighbours']:
                #                 print('N',neighbour_name, self.coloring[neighbour_name] ,)
                available_colors -= {self.coloring[neighbour_name]}
                if len(available_colors) == 0:
                    break
            if len(available_colors) == 0:
                max_color += 1  # color is index in candidates_coloring
                self.used_colors[max_color] = set()
                self.used_colors[max_color].add(vertex_name)
                self.coloring[vertex_name] = max_color

            #             print('available_colors',available_colors)
            if len(available_colors) != 0:
                for available_color in available_colors:
                    rand_available_color = available_color
                    break
                self.used_colors[rand_available_color].add(vertex_name)
                self.coloring[vertex_name] = rand_available_color

    #             print('chosen_Color',self.coloring[vertex_name])

    def recolor_by_degree_reverse(self):
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
        for vertex_index in range(len(self.D) - 1, -1, -1):
            vertex = self.D[vertex_index]
            vertex_name = vertex['name']
            #             print('vertex_name',vertex_name)
            #             print('used_Colors',self.used_colors)
            available_colors = set(self.used_colors.keys())
            for neighbour_name in self.V[vertex_name]['neighbours']:
                #                 print('N',neighbour_name, self.coloring[neighbour_name] ,)
                available_colors -= {self.coloring[neighbour_name]}
                if len(available_colors) == 0:
                    break
            if len(available_colors) == 0:
                max_color += 1  # color is index in candidates_coloring
                self.used_colors[max_color] = set()
                self.used_colors[max_color].add(vertex_name)
                self.coloring[vertex_name] = max_color

            #             print('available_colors',available_colors)
            if len(available_colors) != 0:
                for available_color in available_colors:
                    rand_available_color = available_color
                    break
                self.used_colors[rand_available_color].add(vertex_name)
                self.coloring[vertex_name] = rand_available_color

    #             print('chosen_Color',self.coloring[vertex_name])

    def initial_heuristic_degree(self):
        clique = set()
        self.sort_vertex_by_degree()
        for vertex in self.D:
            #         print(type(clique),type(vertex))
            # all elements in clique are neighbours to vertex
            all_are_neighbours = True
            for element in clique:
                all_are_neighbours = all_are_neighbours and self.is_neighbour(
                    vertex['name'], element)
                if not all_are_neighbours:
                    break
            #         print(vertex['name'],all_are_neighbours )
            if all_are_neighbours:
                clique.add(vertex['name'])
        return len(clique)

    def initial_heuristic_color(self):
        """
        makes clique:
        first - vertices with maximal color
        """
        clique = set()
        self.sort_vertex_by_degree()

        for color in range(len(self.used_colors)):
            for vertex_name in self.used_colors[color]:
                # all elements in clique are neighbours to vertex
                all_are_neighbours = True
                for element_name in clique:
                    all_are_neighbours = all_are_neighbours and self.is_neighbour(
                        vertex_name, element_name)
                    if not all_are_neighbours:
                        break
                #             print(vertex_name,all_are_neighbours )
                if all_are_neighbours:
                    clique.add(vertex_name)
        return len(clique)

    def initial_heuristic_color_reverse(self):
        """
        makes clique:
        first - vertices with maximal color
        """

        clique = set()
        self.sort_vertex_by_degree()

        for color in range(len(self.used_colors) - 1, -1, -1):
            for vertex_name in self.used_colors[color]:
                # all elements in clique are neighbours to vertex
                all_are_neighbours = True
                for element_name in clique:
                    all_are_neighbours = all_are_neighbours and self.is_neighbour(
                        vertex_name, element_name)
                    if not all_are_neighbours:
                        break
                #             print(vertex_name,all_are_neighbours )
                if all_are_neighbours:
                    clique.add(vertex_name)
        return len(clique)

    def find_init_heuristic(self):
        # self.D.sort(key=lambda i: len(i['neighbours']), reverse=True)
        # self.recolor()
        self.recolor()
        c1 = len(self.used_colors)
        self.clear_coloring()

        self.recolor_by_degree()
        c2 = len(self.used_colors)
        self.clear_coloring()

        self.recolor_by_degree_reverse()
        c3 = len(self.used_colors)
        self.clear_coloring()

        if c1 is min(c1, c2, c3):
            self.clear_coloring()
            self.recolor()

        if c2 is min(c1, c2, c3):
            self.recolor_by_degree()
            self.recolor()
        if c3 is min(c1, c2, c3):
            self.recolor_by_degree_reverse()
            self.recolor()

        init_clique = self.initial_heuristic_color()
        init_clique_rev = self.initial_heuristic_color_reverse()
        init_clique_degree = self.initial_heuristic_degree()
        return max(init_clique, init_clique_rev, init_clique_degree)


def read_dimacs_graph(file_path):
    # Parse .col file and return graph object
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


import copy


def find_independent_sets_by_coloring(coloring_colors, vertex_coloring):
    colors = copy.deepcopy(coloring_colors)
    ind_sets_coloring = dict()
    for vertex_name in vertex_coloring:
        ind_sets_coloring[vertex_name] = {vertex_coloring[vertex_name]}
    for color in colors:
        for vertex_name in coloring_colors[color]:
            available_colors = set(colors.keys())
            avalible_neighbours = g.V[vertex_name]['neighbours'].intersection(candidates)
            while len(avalible_neighbours) > 0:
                neigbour = avalible_neighbours.pop()
                neigbour_color = vertex_coloring[neigbour]
                neigbour_colos_set = ind_sets_coloring[neigbour]
                available_colors -= {neigbour_color}.union(neigbour_colos_set)
                for n_color in neigbour_colos_set:
                    avalible_neighbours -= colors[n_color]
            check_neighbours = set()
            # есть цвета по которым может быть пересечение. Их надо проверить
            for available_color in available_colors:
                avalible_neighbours = g.V[vertex_name]['neighbours'].intersection(candidates)
                avalible_neighbours = avalible_neighbours.intersection(colors[available_color])  # change to colors
                check_neighbours = check_neighbours.union(avalible_neighbours)
            for neighbour in check_neighbours:
                available_colors -= ind_sets_coloring[neighbour]
            for avalible_color in available_colors:
                colors[avalible_color].add(vertex_name)
            ind_sets_coloring[vertex_name] = ind_sets_coloring[vertex_name].union(available_colors)
    return colors, ind_sets_coloring
