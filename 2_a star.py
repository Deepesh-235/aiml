class Graph:
    def __init__(self, graph):
        self.graph = graph  # Adjacency list with weights

    # Heuristic function (just gives a simple guess of distance to goal)
    def heuristic(self, node):
        H = {'A': 1, 'B': 1, 'C': 1, 'D': 1}
        return H.get(node, 0)

    def a_star(self, start, goal):
        open_set = set([start])
        came_from = {}  # For storing path
        g_score = {start: 0}

        while open_set:
            current = min(open_set, key=lambda x: g_score[x] + self.heuristic(x))

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                print("Path found:", path)
                return path

            open_set.remove(current)

            for neighbor, cost in self.graph.get(current, []):
                temp_g = g_score[current] + cost
                if neighbor not in g_score or temp_g < g_score[neighbor]:
                    g_score[neighbor] = temp_g
                    came_from[neighbor] = current
                    open_set.add(neighbor)

        print("Path not found!")
        return None

# ---------- Example graph ----------
graph_data = {
    'A': [('B', 1), ('C', 3), ('D', 7)],
    'B': [('D', 5)],
    'C': [('D', 12)]
}

g = Graph(graph_data)
g.a_star('A', 'D')
