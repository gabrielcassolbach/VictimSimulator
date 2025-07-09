import os
import random
import numpy as np
import time
import joblib
import csv
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sequencer import Sequencer
from collections import deque
from pathlib import Path


class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers, cluster=[]):
        super().__init__(env, config_file)
        self.nb_of_explorers = nb_of_explorers
        self.received_maps = 0
        self.map = Map() 
        self.walk_constant = 1.525
        self.all_victims = {} 
        self.plan = []              
        self.plan_x = 0             
        self.plan_y = 0             
        self.plan_visited = set()   
        self.plan_rtime = self.TLIM 
        self.plan_walk_time = 0.0   
        self.cluster = cluster
        self.rescue_plan = {}
        self.x = 0                  
        self.y = 0                  
        self.set_state(VS.IDLE)

    _SEVERITY_CLF_PATH = Path("models/severity_clf_mlp.joblib")
    _SEVERITY_REG_PATH = Path("models/severity_reg.joblib")
    
    def _load_or_train_models(self, X_train = None, y_class = None, y_value = None):
        if self._SEVERITY_CLF_PATH.exists() and self._SEVERITY_REG_PATH.exists():
            self.classifier = joblib.load(self._SEVERITY_CLF_PATH)
            self.regressor  = joblib.load(self._SEVERITY_REG_PATH)
            return

    def predict_severity_and_class(self):
        if not hasattr(self, "classifier") or not hasattr(self, "regressor"):
            raise RuntimeError("Modelos não foram carregados/treinados.")

        with open("output/victim_data.txt", "w", encoding="utf-8") as f:
            for vic_id, (coords, vitals) in self.all_victims.items():
                X = np.array(vitals[1:6]).reshape(1, -1)
                severity_class = int(self.classifier.predict(X)[0])
                severity_value = float(self.regressor.predict(X)[0])
                f.write(f"{vic_id}, {coords[0]}, {coords[1]}, {severity_value}, {severity_class}\n")        
                vitals.extend([severity_value, severity_class])

    def save_cluster_csv(self, cluster_id, cluster):
        filename = f"clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, coord, signals in cluster:
                x, y = coord    
                vs = signals       
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, coord, signals in sequence:
                x, y = coord     
                vs = signals     
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def cluster_victims(self, victims_data, n_clusters=4, random_state=42):
        victims_positions = [coord for _, coord, _ in victims_data]
        X = np.array(victims_positions)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        clustered_data = {i: [] for i in range(n_clusters)}
        for i, (vic_id, coord, signals) in enumerate(victims_data):
            cluster_label = labels[i]
            clustered_data[cluster_label].append((vic_id, coord, signals))
    
        return clustered_data

    def sync_explorers(self, explorer_map, victims):
        self.received_maps += 1
        self.map.update(explorer_map)
        self.all_victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            victims_data = [
                (vic_id, coord, vital_signals)
                for vic_id, (coord, vital_signals) in self.all_victims.items()
            ]

            clusters = self.cluster_victims(victims_data)
            self._load_or_train_models()
            self.predict_severity_and_class()

            for it in range(0, 4):
                self.save_cluster_csv(it + 1, clusters[it])

            rescuers = [None] * 4
            rescuers[0] = self 
            self.cluster = clusters[0]

            for i in range(1, 4):
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, clusters[i]) 
                rescuers[i].map = self.map   
                
            for i, rescuer in enumerate(rescuers):
                rescuer.victims_rescue_seq()
                self.save_sequence_csv(rescuers[i].rescue_plan, i+1)             
                rescuer.planner()
                rescuer.set_state(VS.ACTIVE)

    def victims_rescue_seq(self):
        rescue_plan = {}

        id_map      = {coord: vic_id      for vic_id, coord, _       in self.cluster}
        signals_map = {coord: signals    for _,      coord, signals in self.cluster}
        victims = [coord for _, coord, _ in self.cluster]

        if len(victims) == 1:
            rescue_plan = victims  
        else:
            seq = Sequencer()
            bfs_route = seq.bfs_like_sequence(victims)
            best_route = seq.two_opt(bfs_route)
            rescue_plan = best_route
            
        self.rescue_plan = [
            ( id_map[coord],      
            coord,              
            signals_map[coord]  
            )
            for coord in best_route
        ]

    def get_neighbors(self, node):
        neighbors = []

        for direction in range(8):
            dx, dy = AbstAgent.AC_INCR[direction]
            coord = (node[0] + dx, node[1] + dy)
            if self.map.in_map(coord):
                neighbors.append(coord)

        return neighbors

    def calculatepath_tovictim(self, start, goal):
        queue = deque([start]) 
        visited = set([start]) 
        parent = {}

        while queue: 
            current = queue.popleft()

            if current == goal:
                path = []
                while current != start:
                    prev = parent[current]
                    dx = current[0] - prev[0]
                    dy = current[1] - prev[1]
                    path.append((dx, dy, False))
                    current = prev
                path.reverse()
                if goal[0] != 0 and goal[1] != 0:
                    dx, dy, has_victim = path[-1]
                    path[-1] = (dx, dy, True)
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        return {}

        
    def planner(self):
        prev_goal = (0, 0)
        all_plans = []
        total_time = 0
       
        for _, coord, _ in self.rescue_plan:
            new_plan = self.calculatepath_tovictim(prev_goal, coord)
            all_plans.append(new_plan)
            total_time += len(new_plan)*self.walk_constant
            prev_goal = coord

        return_plan = self.calculatepath_tovictim(prev_goal, (0, 0))
        
        while total_time + len(return_plan)*self.walk_constant >= self.TLIM and len(all_plans) > 0:
            total_time -= len(all_plans.pop())*self.walk_constant
            it = len(all_plans) - 1
            return_plan = self.calculatepath_tovictim(self.rescue_plan[it][1], (0, 0))
            
        if len(all_plans) > 0:
            all_plans.append(return_plan)
        
        for plan in all_plans:
            self.plan.extend(plan)

    def deliberate(self) -> bool:
        if self.plan == []:
           return False

        dx, dy, there_is_vict = self.plan.pop(0)
        walked = self.walk(dx, dy)

        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            
            if there_is_vict:
                rescued = self.first_aid() 
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")

        return True


