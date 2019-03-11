#!/usr/bin/env python
import numpy as np
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class AStar():
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    def __init__(self):
        self.pre_path = []

    def initial(self, maze, start, end):
        # Create start and end node
        self.start_node = Node(None, start)
        self.end_node = Node(None, end)

        # Initialize both open and closed list
        self.open_list = []
        self.closed_list = []

        # Add the start node
        self.open_list.append(self.start_node)
        
        # Define map
        self.maze = maze
        h, w = self.maze.shape
        self.h_w = min(h, w)
        
    def planning(self):
        # Loop until you find the end
        count = 0
        while len(self.open_list) > 0:
            count = count + 1
            if count > self.h_w/2:
                print("A* failed to find a solution")
                return self.pre_path, False

            # Get the current node
            current_node = self.open_list[0]
            current_index = 0
            for index, item in enumerate(self.open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            self.open_list.pop(current_index)
            self.closed_list.append(current_node)

            # Found the goal
            if current_node == self.end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                self.pre_path = path[::-1]
                return path[::-1], True # Return reversed path

            # Generate children
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(self.maze) - 1) or node_position[0] < 0 or node_position[1] > (len(self.maze[len(self.maze)-1]) -1) or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if self.maze[node_position[0]][node_position[1]] == 100.:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                for closed_child in self.closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - self.end_node.position[0]) ** 2) + ((child.position[1] - self.end_node.position[1]) ** 2)
                child.f = child.g + child.h + 5*self.maze[child.position[0]][child.position[1]]

                # Child is already in the open list
                for open_node in self.open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                self.open_list.append(child)
        