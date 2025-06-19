import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from collections import deque
import time
from map import Map

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        super().__init__(env, config_file)
        self.set_state(VS.ACTIVE) 
        self.resc = resc
        self.total_exploration_time = 0
        self.worst_move_scenario = 0
        self.exploration_flag = True
        self.x = 0  
        self.y = 0 
        self.path = []
        self.map = Map() 
        self.victims = {} 
        self.visited = set()
        self.return_path = {}
        self.path_it = 0
        self.visited.add((self.x, self.y))

        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def look_around(self):
        obstacles = self.check_walls_and_lim()
        neighbors = []

        for direction in range(8):
            dx, dy = AbstAgent.AC_INCR[direction]
            new_pos = (self.x + dx, self.y + dy)
            if obstacles[direction] == VS.CLEAR and new_pos not in self.visited:
                neighbors.append((dx, dy))

        return [(dx, dy) for dx, dy in neighbors]
            
    def update_coordinates(self, dx, dy):
        self.x += dx
        self.y += dy
        self.visited.add((self.x, self.y))
        
    def check_for_victims(self, dx, dy, rtime_bef, rtime_aft):
        seq_victim = self.check_for_victim()
        if seq_victim != VS.NO_VICTIM:
            vs = self.read_vital_signals()
            self.victims[vs[0]] = ((self.x, self.y), vs)
            
        difficulty = (rtime_bef - rtime_aft)
        difficulty = (difficulty / self.COST_LINE) if dx == 0 or dy == 0 else (difficulty / self.COST_DIAG)

        self.map.add((self.x, self.y), difficulty, seq_victim, self.check_walls_and_lim())

    def get_neighbors(self, node):
        neighbors = []

        for direction in range(8):
            dx, dy = AbstAgent.AC_INCR[direction]
            coord = (node[0] + dx, node[1] + dy)
            if self.map.in_map(coord):
                neighbors.append(coord)

        return neighbors

    def compute_path_to_base(self):
        if self.x == 0 and self.y == 0:
            return {}

        start = (self.x, self.y)
        queue = deque([start]) 
        visited = set([start]) 

        goal = (0, 0)
        parent = {}

        while queue: 
            current = queue.popleft()

            if current == goal:
                path = []
                while current != start:
                    prev = parent[current]
                    dx = current[0] - prev[0]
                    dy = current[1] - prev[1]
                    path.append((dx, dy))
                    current = prev
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        return {}
                
    def explore(self):
        print("exploring ....")
        neighbors = self.look_around()

        if neighbors:
            dx, dy = neighbors[0]
            self.path.append((self.x, self.y)) 

            rtime_bef = self.get_rtime()
            result = self.walk(dx, dy)
            rtime_aft = self.get_rtime()

            #self.total_exploration_time += (rtime_bef - rtime_aft) 
            self.worst_move_scenario = max(self.worst_move_scenario, rtime_bef - rtime_aft)           

            if result == VS.EXECUTED:
                self.update_coordinates(dx, dy)
                self.check_for_victims(dx, dy, rtime_bef, rtime_aft)
        else:
            self.backtrack()

    def backtrack(self):
        while self.path:
            prev_x, prev_y = self.path.pop()
            dx = prev_x - self.x 
            dy = prev_y - self.y 

            rtime_bef = self.get_rtime()
            result = self.walk(dx, dy)
            rtime_aft = self.get_rtime()

            #self.total_exploration_time += (rtime_aft - rtime_bef) 

            if result == VS.EXECUTED:
                self.update_coordinates(dx, dy)
                return
    
    def estimate_return_time(self):
        path = self.compute_path_to_base()
        return path, len(path)*self.worst_move_scenario

    def can_explore(self):
        path, return_time = self.estimate_return_time()

        if return_time + 10 >= self.get_rtime():
            self.exploration_flag = False
            return path

        return path
        
    def returnto_base(self):  
        print("returning to base....")
        next_value = self.return_path[self.path_it]
        self.path_it += 1
        self.walk(next_value[0], next_value[1])
        self.x += next_value[0]
        self.y += next_value[1]
    
    def deliberate(self) -> bool:
        print("remanining time: ", self.get_rtime())
        if self.exploration_flag: 
            self.explore()
            self.return_path = self.can_explore()
        else:
            self.returnto_base()
            
        if self.x == 0 and self.y == 0:
            self.resc.sync_explorers(self.map, self.victims)
            return False

        return True

