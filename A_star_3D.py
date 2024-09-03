"""
 Copyright (c) 2024, Your Name
 All rights reserved.
 CopyrightText: SEO HYEON HO
 License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 """

import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pandas as pd

def a_star_3d(start, goal, grid, obstacles, obstacle_radius, agent_radius):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def is_collision(node, obstacles, obstacle_radius, agent_radius):
        for obstacle in obstacles:
            if np.linalg.norm(np.array(node) - np.array(obstacle)) <= (obstacle_radius + agent_radius):
                return True
        return False

    def get_neighbors(node, previous_node=None):
        neighbors = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if (dx, dy, dz) != (0,0,0)]
        result = []
        for dx, dy, dz in neighbors:
            x2, y2, z2 = node[0] + dx, node[1] + dy, node[2] + dz
            if 0 <= x2 < grid.shape[0] and 0 <= y2 < grid.shape[1] and 0 <= z2 < grid.shape[2] and not is_collision((x2, y2, z2), obstacles, obstacle_radius, agent_radius):
                result.append((x2, y2, z2))
        
        if previous_node:
            result.sort(key=lambda n: np.arccos(np.dot(np.array(n) - np.array(previous_node), np.array(goal) - np.array(previous_node)) / 
                                               (np.linalg.norm(np.array(n) - np.array(previous_node)) * np.linalg.norm(np.array(goal) - np.array(previous_node)))))
        
        return result

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current_g, current, previous_node = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, previous_node):
            tentative_g_score = current_g + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor, current))

    return None

def interpolate_path(path, max_speed):
    interpolated_path = []
    for i in range(len(path) - 1):
        start = np.array(path[i])
        end = np.array(path[i + 1])
        distance = np.linalg.norm(end - start)
        steps = int(np.ceil(distance / max_speed)) 
        for step in range(steps):
            interpolated_path.append(start + (end - start) * (step / steps))
    interpolated_path.append(np.array(path[-1]))
    return interpolated_path

def update(frame):
    ax.cla()
    
    for agent_path in agent_paths:
        if frame < len(agent_path):
            x, y, z = agent_path[frame]
            ax.scatter(x, y, z, s=100)
        ax.plot([x for (x, y, z) in agent_path], [y for (x, y, z) in agent_path], [z for (x, y, z) in agent_path], '-o')
    
    for index, row in combined.iterrows():
        start = row['agents']
        goal = row['goals']
        ax.scatter(start[0], start[1], start[2], c='g', s=100, label='Start' if index == 0 else "")
        ax.scatter(goal[0], goal[1], goal[2], c='r', s=100, marker='x', label='Goal' if index == 0 else "")
    
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    for obstacle in obstacles:
        x = obstacle[0] + obstacle_radius * np.outer(np.cos(u), np.sin(v))
        y = obstacle[1] + obstacle_radius * np.outer(np.sin(u), np.sin(v))
        z = obstacle[2] + obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='r', alpha=0.2)

    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_zlim(0, grid.shape[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Pathfinding with A* Algorithm")
    ax.legend()

def setup_scenario():
    agents = [(0, 100, 100), (0, 0, 100), (0, 100, 0), (0, 0, 0)]
    goals = [(100, 0, 0), (100, 100, 0), (100, 0, 100), (100, 100, 100)]

    combined = pd.DataFrame({
        'agents': agents,
        'goals': goals
    })

    def calculate_distance(row):
        return np.linalg.norm(np.array(row['agents']) - np.array(row['goals']))

    combined['distance'] = combined.apply(calculate_distance, axis=1)

    combined = combined.sort_values('distance', ascending=False)

    combined = combined.reset_index(drop=True)

    return combined

def main():
    global path, combined, grid, obstacles, obstacle_radius, ax, agent_paths, agent_radius
    grid_size = 101
    grid = np.zeros((grid_size, grid_size, grid_size))
    
    agent_radius = 3 
    obstacle_radius = agent_radius  

    obstacles = [(random.uniform(40, 60), random.uniform(40, 60), random.uniform(40, 60)) for _ in range(10)]
    leng_obstacles = len(obstacles)
    combined = setup_scenario()
    
    agent_paths = []
    
    for index, row in combined.iterrows():
        start = row['agents']
        goal = row['goals']
        path = a_star_3d(start, goal, grid, obstacles, obstacle_radius, agent_radius)
        if path is None:
            print(f"경로를 찾을 수 없습니다. 시작: {start}, 목표: {goal}")
            return
        interpolated_path = interpolate_path(path, max_speed=10.5)
        agent_paths.append(interpolated_path)
        
        for point in interpolated_path:
            obstacles.append(tuple(point))
    obstacles = obstacles[:leng_obstacles]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    max_path_length = max(len(agent_path) for agent_path in agent_paths)
    anim = FuncAnimation(fig, update, frames=max_path_length, interval=100, repeat=False)
    plt.show()

if __name__ == '__main__':
    main()