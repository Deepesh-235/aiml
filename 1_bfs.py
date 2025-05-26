from collections import deque

class Graph:
    def __init__(self):
        # Dictionary to store the graph (adjacency list)
        self.graph = {}

    def add_edge(self, u, v):
        # Add an edge from u to v
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def bfs(self, start):
        visited = set()         # To track visited nodes
        queue = deque([start])  # Use deque for efficient queue operations
        visited.add(start)

        while queue:
            node = queue.popleft()
            print(node, end=" ")

            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

# ---------- Test the code ----------

g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

print("Breadth First Traversal (starting from vertex 2):")
g.bfs(2)
