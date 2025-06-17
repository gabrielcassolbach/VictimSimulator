import os
import random
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import numpy as np

class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        super().__init__(env, config_file)
        self.map = None             
        self.victims = None         
        self.plan = []              
        self.plan_x = 0             
        self.plan_y = 0             
        self.plan_visited = set()   
        self.plan_rtime = self.TLIM 
        self.plan_walk_time = 0.0   
        self.victims_group = []
        self.x = 0                  
        self.y = 0                  
        self.set_state(VS.IDLE)

    def cluster_victims(self, victims_positions, n_clusters=4, random_state=42):
        X = np.array(victims_positions)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        return labels, centroids

    def go_save_victims(self, map, victims):
        print(f"\n\n*** R E S C U E R ***")
        self.map = map
        print(f"{self.NAME} Map received from the explorer")
        self.victims = victims

        victims_positions = [coord for coord, _ in victims.values()]

        self.__planner(victims_positions)       
        self.set_state(VS.ACTIVE)
    
    def divide_victims(self, victims_positions):
        labels, _ = self.cluster_victims(victims_positions)
        self.victims_group = list(zip(victims_positions, labels.tolist()))

    def __planner(self, victims_positions):
        self.divide_victims(victims_positions)
        
        # 1 - Subdividir as vítimas encontradas em grupos. 
        #----> implementar algoritmo de clustering. 
        # 2 - Definir a sequência de salvamento das vítimas. 
        #----> implementar algoritmo genético. 
        # 3 - Definir, utilizando um algortimo de busca, um caminho entre 
        # as vítimas. A*, por exemplo.
        

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

