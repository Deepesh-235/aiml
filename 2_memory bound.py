import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Maze:
    def __init__(self, size=25):
        self.size = size
        self.grid = np.zeros((size, size), dtype=bool)
        self.generate()

    def generate(self):
        stack = deque([(1, 1)])
        while stack:
            x, y = stack.pop()
            self.grid[x, y] = 1
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx * 2, y + dy * 2
                if 0 < nx < self.size and 0 < ny < self.size and not self.grid[nx, ny]:
                    stack.append((nx, ny))
                    self.grid[nx, ny] = 1

def astar(graph, start, goal):
    open_set = {start}
    came_from = {}
    gscore = {start: 0}
    while open_set:
        current = min(open_set, key=lambda node: gscore[node] + abs(goal[0] - node[0]) + abs(goal[1] - node[1]))
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        open_set.remove(current)
        for neighbor in graph.get(current, []):
            tentative_gscore = gscore[current] + 1
            if tentative_gscore < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                open_set.add(neighbor)
    return None

def plot(maze, path):
    plt.imshow(maze, cmap='binary')
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], color='red', linewidth=2)
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    size = int(input("Enter size of maze: "))
    maze = Maze(size).grid
    start, goal = (1, 1), (size-2, size-2)
    graph = {(x, y): [(x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] if maze[x + dx, y + dy]] for x in range(1, size - 1) for y in range(1, size - 1)}
    shortest_path = astar(graph, start, goal)
    plot(maze, shortest_path)
15
