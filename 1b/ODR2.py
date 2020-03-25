##Ten local community networks with a population size of 50 agents were generated using
##the random graph generator, random_regular_graph.
##Agents learn using Q-Learning algorithm (exploration rate: 0.1, learning rate: 0.4).
##Number of actions = 4 (action 0, 1, 2, 3).
##Conformity is the average conformity of all the ten local communities. 
##Off-Diagonal Scenario #2


import numpy as np
#import matplotlib.pyplot as plt
import random
import secrets
import pickle
import networkx as nx
import json
#from matplotlib.backends.backend_pdf import PdfPages
from os import listdir
import math
      
class Simulation:
    def __init__(self, graph):
        self.graph = graph
        self.num_agents = graph.number_of_nodes() # number_of_nodes() - returns the number of nodes in the graph

        # number of actions
        self.n_actions = 4
        # total number of iterations
        self.number_of_iterations = 2000
        
        self.cumulative_reward_sum_array = []  # array to store cumulative reward sum after each iteration 

        self.norm1_frequency_array_lc1 = [] # array to store frequency of joint action 0 and 0 by both agents after each iteration
        self.norm1_frequency_array_lc2 = [] 
        self.norm1_frequency_array_lc3 = [] 
        self.norm1_frequency_array_lc4 = [] 
        self.norm1_frequency_array_lc5 = [] 
        self.norm1_frequency_array_lc6 = [] 
        self.norm1_frequency_array_lc7 = [] 
        self.norm1_frequency_array_lc8 = [] 
        self.norm1_frequency_array_lc9 = [] 
        self.norm1_frequency_array_lc10 = [] 
        
        self.norm2_frequency_array_lc1 = []
        self.norm2_frequency_array_lc2 = [] 
        self.norm2_frequency_array_lc3 = []
        self.norm2_frequency_array_lc4 = []
        self.norm2_frequency_array_lc5 = []
        self.norm2_frequency_array_lc6 = []
        self.norm2_frequency_array_lc7 = []
        self.norm2_frequency_array_lc8 = []
        self.norm2_frequency_array_lc9 = []
        self.norm2_frequency_array_lc10 = [] 

        self.norm3_frequency_array_lc1 = [] # array to store frequency of joint action 0 and 0 by both agents after each iteration
        self.norm3_frequency_array_lc2 = [] 
        self.norm3_frequency_array_lc3 = [] 
        self.norm3_frequency_array_lc4 = [] 
        self.norm3_frequency_array_lc5 = [] 
        self.norm3_frequency_array_lc6 = [] 
        self.norm3_frequency_array_lc7 = [] 
        self.norm3_frequency_array_lc8 = [] 
        self.norm3_frequency_array_lc9 = [] 
        self.norm3_frequency_array_lc10 = [] 
        
        self.norm4_frequency_array_lc1 = []
        self.norm4_frequency_array_lc2 = [] 
        self.norm4_frequency_array_lc3 = []
        self.norm4_frequency_array_lc4 = []
        self.norm4_frequency_array_lc5 = []
        self.norm4_frequency_array_lc6 = []
        self.norm4_frequency_array_lc7 = []
        self.norm4_frequency_array_lc8 = []
        self.norm4_frequency_array_lc9 = []
        self.norm4_frequency_array_lc10 = [] 


        self.diversity_array = [] # array to store diversity after each iteration
        #self.conformity_array = []
        self.conformity_array1 = [] # array to store conformity of a norm after each iteration
        self.conformity_array2 = []
        self.conformity_array3 = []
        self.conformity_array4 = []
        self.conformity_array5 = []
        self.conformity_array6 = []
        self.conformity_array7 = []
        self.conformity_array8 = []
        self.conformity_array9 = []
        self.conformity_array10 = []
        
        self.cumulative_reward_sum = 0 # cumulative reward sum in an iteration
        self.total_reward = np.zeros(self.number_of_iterations) # total reward at a particular iteration
        self.average_reward = np.zeros(self.number_of_iterations) # average reward at a particular iteration
        self.norms_cumulative = np.zeros(self.number_of_iterations) # norms cumulative at a particular iteration

        agent_id = 1 # initializes unique agent id

        # creates a new agent for each node, by creating an agents_map and then relabeling the nodes in the graph
        self.agents_map = {}
        for i in range(0, self.num_agents):
            self.agents_map[i] = Agent(agent_id, alpha, gamma, epsilon)
            #print ("agent_id: ", agent_id)
            agent_id = agent_id + 1
        #nx.ODR2aw_networkx(self.graph,with_labels=True,node_size=500)
        #plt.show()
        #for n in self.graph.nodes():
            #print("before relabel nodes...")
            #print (n)

        self.graph = nx.convert_node_labels_to_integers(self.graph)
        
        nx.relabel_nodes(self.graph, self.agents_map, copy=False)
        #i = 1
        #for n in self.graph.nodes():
            #print("after relabel nodes...")
            #print (n)
        #nx.draw_networkx(self.graph,with_labels=True,node_size=500)
        #plt.show()

        #Relabel the nodes of the graph G.
        #Parameters:	
        #G (graph) – A NetworkX graph
        #mapping (dictionary) – A dictionary with the old labels as keys and new labels as values. A partial mapping is allowed.
        #copy (bool (optional, default=True)) – If True return a copy, or if False relabel the nodes in place.
        #https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.relabel.relabel_nodes.html
        

    def simulate(self):
        # cache each agent's neighbor list - could looked up each time depending what you are doing

        # In Python source code, you sometimes see a for loop inside a square bracket([]).
        # It is called List comprehension.
        # It is a way to create a new list from an old list. It has the following syntax.

        #new_list = [ NEW_VALUE for item in YOUR_LIST ]
        #You should read it as: For item in the old list, assign the new value to NEW_VALUE.
        #https://openwritings.net/pg/python/python-loop-square-brackets


        for n in self.graph.nodes():            
            n.neighbors = [agt for agt in self.graph.neighbors(n)]

        conformity_in_iteration = 0
        diversity_in_iteration = 0
        normalization_constant = 1/(math.log2(4)) # as there are four actions, cardinality is 4; for 3 it should be log2(3)


        attrs_graph = nx.get_node_attributes(self.graph, 'LN')

        for iteration in range(self.number_of_iterations):
            counter = 0
            reward_total_in_the_iteration = 0
            norm1_count = 0
            norm2_count = 0
            norms_cumulative_count = 0

            a0c_lc1 = 0 # action 0 count in each iteration to calculate conformity at each iteration / norm1 count in lc1
            a1c_lc1 = 0 # action 1 count in each iteration to calculate conformity at each iteration / norm2 count in lc1
            a2c_lc1 = 0
            a3c_lc1 = 0
            
            a0c_lc2 = 0
            a1c_lc2 = 0
            a2c_lc2 = 0
            a3c_lc2 = 0
            
            a0c_lc3 = 0
            a1c_lc3 = 0
            a2c_lc3 = 0
            a3c_lc3 = 0
            
            a0c_lc4 = 0
            a1c_lc4 = 0
            a2c_lc4 = 0
            a3c_lc4 = 0
            
            a0c_lc5 = 0
            a1c_lc5 = 0
            a2c_lc5 = 0
            a3c_lc5 = 0
            
            a0c_lc6 = 0
            a1c_lc6 = 0
            a2c_lc6 = 0
            a3c_lc6 = 0
            
            a0c_lc7 = 0
            a1c_lc7 = 0
            a2c_lc7 = 0
            a3c_lc7 = 0
            
            a0c_lc8 = 0
            a1c_lc8 = 0
            a2c_lc8 = 0
            a3c_lc8 = 0
            
            a0c_lc9 = 0
            a1c_lc9 = 0
            a2c_lc9 = 0
            a3c_lc9 = 0
            
            a0c_lc10 = 0
            a1c_lc10 = 0
            a2c_lc10 = 0
            a3c_lc10 = 0

            norm1_count_lc1 = 0
            norm2_count_lc1 = 0
            norm3_count_lc1 = 0
            norm4_count_lc1 = 0
            
            norm1_count_lc2 = 0
            norm2_count_lc2 = 0
            norm3_count_lc2 = 0
            norm4_count_lc2 = 0
            
            norm1_count_lc3 = 0
            norm2_count_lc3 = 0
            norm3_count_lc3 = 0
            norm4_count_lc3 = 0
            
            norm1_count_lc4 = 0
            norm2_count_lc4 = 0
            norm3_count_lc4 = 0
            norm4_count_lc4 = 0
            
            norm1_count_lc5 = 0
            norm2_count_lc5 = 0
            norm3_count_lc5 = 0
            norm4_count_lc5 = 0
            
            norm1_count_lc6 = 0
            norm2_count_lc6 = 0
            norm3_count_lc6 = 0
            norm4_count_lc6 = 0
            
            norm1_count_lc7 = 0
            norm2_count_lc7 = 0
            norm3_count_lc7 = 0
            norm4_count_lc7 = 0
            
            norm1_count_lc8 = 0
            norm2_count_lc8 = 0
            norm3_count_lc8 = 0
            norm4_count_lc8 = 0

            norm1_count_lc9 = 0
            norm2_count_lc9 = 0
            norm3_count_lc9 = 0
            norm4_count_lc9 = 0
            
            norm1_count_lc10 = 0
            norm2_count_lc10 = 0
            norm3_count_lc10 = 0
            norm4_count_lc10 = 0
            #atc = 0 # total action count to calculate conformity at each iteration

            for x in self.graph.nodes():
                #print ("-----------------------------")
                #print("x.neighbors is: ", x.neighbors)
                    
                z = random.choice(x.neighbors)
                #print ("x: ", x, " - ", "z: ", z)
                #print("x.agentid", x.agentid, " selected partner / z.agentid ", z.agentid)
                #print ("counter: ", counter, " - z.agentid-1", z.agentid-1)
                #print ("-----------------------------")

                #print("Printing node attribute of a node: ", attrs_graph[x])
                #if ('LN1' in attrs_graph[x]):
                    #print ("inside simulate(self) for G1")

                agent1_action = self.agents_map[counter].choose_action(self.agents_map[counter].QTable, epsilon, self.n_actions)
                #print ("self.agents_map[counter]: ", self.agents_map[counter], " - ", "agent1_action: ", agent1_action)
                agent2_action = self.agents_map[z.agentid-1].choose_action(self.agents_map[z.agentid-1].QTable, epsilon, self.n_actions)
                #print ("self.agents_map[z.agentid-1]: ", self.agents_map[z.agentid-1], " - ", "agent2_action: ", agent2_action)

                if(agent1_action == 0 and agent2_action == 0):
                    norm1_count += 1
                    norms_cumulative_count += 1
                elif(agent1_action == 1 and agent2_action == 1):
                    norm2_count += 1
                    norms_cumulative_count += 1
                # --------
                if((('LN1' in attrs_graph[x]) and agent1_action == 0) and (('LN1' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc1 += 1
                if((('LN1' in attrs_graph[x]) and agent1_action == 1) and (('LN1' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc1 += 1
                if((('LN1' in attrs_graph[x]) and agent1_action == 2) and (('LN1' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc1 += 1
                if((('LN1' in attrs_graph[x]) and agent1_action == 3) and (('LN1' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc1 += 1
                   
                if((('LN2' in attrs_graph[x]) and agent1_action == 0) and (('LN2' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc2 += 1
                if((('LN2' in attrs_graph[x]) and agent1_action == 1) and (('LN2' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc2 += 1
                if((('LN2' in attrs_graph[x]) and agent1_action == 2) and (('LN2' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc2 += 1
                if((('LN2' in attrs_graph[x]) and agent1_action == 3) and (('LN2' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc2 += 1

                if((('LN3' in attrs_graph[x]) and agent1_action == 0) and (('LN3' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc3 += 1
                if((('LN3' in attrs_graph[x]) and agent1_action == 1) and (('LN3' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc3 += 1
                if((('LN3' in attrs_graph[x]) and agent1_action == 2) and (('LN3' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc3 += 1
                if((('LN3' in attrs_graph[x]) and agent1_action == 3) and (('LN3' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc3 += 1
                   
                if((('LN4' in attrs_graph[x]) and agent1_action == 0) and (('LN4' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc4 += 1
                if((('LN4' in attrs_graph[x]) and agent1_action == 1) and (('LN4' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc4 += 1
                if((('LN4' in attrs_graph[x]) and agent1_action == 2) and (('LN4' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc4 += 1
                if((('LN4' in attrs_graph[x]) and agent1_action == 3) and (('LN4' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc4 += 1

                if((('LN5' in attrs_graph[x]) and agent1_action == 0) and (('LN5' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc5 += 1
                if((('LN5' in attrs_graph[x]) and agent1_action == 1) and (('LN5' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc5 += 1
                if((('LN5' in attrs_graph[x]) and agent1_action == 2) and (('LN5' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc5 += 1
                if((('LN5' in attrs_graph[x]) and agent1_action == 3) and (('LN5' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc5 += 1
                   
                if((('LN6' in attrs_graph[x]) and agent1_action == 0) and (('LN6' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc6 += 1
                if((('LN6' in attrs_graph[x]) and agent1_action == 1) and (('LN6' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc6 += 1
                if((('LN6' in attrs_graph[x]) and agent1_action == 2) and (('LN6' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc6 += 1
                if((('LN6' in attrs_graph[x]) and agent1_action == 3) and (('LN6' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc6 += 1

                if((('LN7' in attrs_graph[x]) and agent1_action == 0) and (('LN7' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc7 += 1
                if((('LN7' in attrs_graph[x]) and agent1_action == 1) and (('LN7' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc7 += 1
                if((('LN7' in attrs_graph[x]) and agent1_action == 2) and (('LN7' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc7 += 1
                if((('LN7' in attrs_graph[x]) and agent1_action == 3) and (('LN7' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc7 += 1
                   
                if((('LN8' in attrs_graph[x]) and agent1_action == 0) and (('LN8' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc8 += 1
                if((('LN8' in attrs_graph[x]) and agent1_action == 1) and (('LN8' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc8 += 1
                if((('LN8' in attrs_graph[x]) and agent1_action == 2) and (('LN8' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc8 += 1
                if((('LN8' in attrs_graph[x]) and agent1_action == 3) and (('LN8' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc8 += 1

                if((('LN9' in attrs_graph[x]) and agent1_action == 0) and (('LN9' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc9 += 1
                if((('LN9' in attrs_graph[x]) and agent1_action == 1) and (('LN9' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc9 += 1
                if((('LN9' in attrs_graph[x]) and agent1_action == 2) and (('LN9' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc9 += 1
                if((('LN9' in attrs_graph[x]) and agent1_action == 3) and (('LN9' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc9 += 1
                   
                if((('LN10' in attrs_graph[x]) and agent1_action == 0) and (('LN10' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc10 += 1
                if((('LN10' in attrs_graph[x]) and agent1_action == 1) and (('LN10' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc10 += 1
                if((('LN10' in attrs_graph[x]) and agent1_action == 2) and (('LN10' in attrs_graph[z]) and agent2_action == 2)):
                   norm3_count_lc10 += 1
                if((('LN10' in attrs_graph[x]) and agent1_action == 3) and (('LN10' in attrs_graph[z]) and agent2_action == 3)):
                   norm4_count_lc10 += 1
                # -----------------------------------------------





                # ----------------------------------------------
                    
                if(('LN1' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc1 += 1
                if(('LN2' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc2 += 1
                if(('LN3' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc3 += 1
                if(('LN4' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc4 += 1
                if(('LN5' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc5 += 1
                if(('LN6' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc6 += 1
                if(('LN7' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc7 += 1
                if(('LN8' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc8 += 1
                if(('LN9' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc9 += 1
                if(('LN10' in attrs_graph[x]) and agent1_action == 0):
                    a0c_lc10 += 1
                    
                if(('LN1' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc1 += 1
                if(('LN2' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc2 += 1
                if(('LN3' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc3 += 1
                if(('LN4' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc4 += 1
                if(('LN5' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc5 += 1
                if(('LN6' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc6 += 1
                if(('LN7' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc7 += 1
                if(('LN8' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc8 += 1
                if(('LN9' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc9 += 1
                if(('LN10' in attrs_graph[x]) and agent1_action == 1):
                    a1c_lc10 += 1

                if(('LN1' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc1 += 1
                if(('LN2' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc2 += 1
                if(('LN3' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc3 += 1
                if(('LN4' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc4 += 1
                if(('LN5' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc5 += 1
                if(('LN6' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc6 += 1
                if(('LN7' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc7 += 1
                if(('LN8' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc8 += 1
                if(('LN9' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc9 += 1
                if(('LN10' in attrs_graph[x]) and agent1_action == 2):
                    a2c_lc10 += 1

                if(('LN1' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc1 += 1
                if(('LN2' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc2 += 1
                if(('LN3' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc3 += 1
                if(('LN4' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc4 += 1
                if(('LN5' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc5 += 1
                if(('LN6' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc6 += 1
                if(('LN7' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc7 += 1
                if(('LN8' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc8 += 1
                if(('LN9' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc9 += 1
                if(('LN10' in attrs_graph[x]) and agent1_action == 3):
                    a3c_lc10 += 1

                    
                if(('LN1' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc1 += 1
                if(('LN2' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc2 += 1
                if(('LN3' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc3 += 1
                if(('LN4' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc4 += 1
                if(('LN5' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc5 += 1
                if(('LN6' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc6 += 1
                if(('LN7' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc7 += 1
                if(('LN8' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc8 += 1
                if(('LN9' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc9 += 1                    
                if(('LN10' in attrs_graph[z]) and agent2_action == 0):
                    a0c_lc10 += 1
                    
                if(('LN1' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc1 += 1
                if(('LN2' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc2 += 1
                if(('LN3' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc3 += 1
                if(('LN4' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc4 += 1
                if(('LN5' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc5 += 1
                if(('LN6' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc6 += 1
                if(('LN7' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc7 += 1
                if(('LN8' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc8 += 1
                if(('LN9' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc9 += 1
                if(('LN10' in attrs_graph[z]) and agent2_action == 1):
                    a1c_lc10 += 1

                if(('LN1' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc1 += 1
                if(('LN2' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc2 += 1
                if(('LN3' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc3 += 1
                if(('LN4' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc4 += 1
                if(('LN5' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc5 += 1
                if(('LN6' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc6 += 1
                if(('LN7' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc7 += 1
                if(('LN8' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc8 += 1
                if(('LN9' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc9 += 1
                if(('LN10' in attrs_graph[z]) and agent2_action == 2):
                    a2c_lc10 += 1

                if(('LN1' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc1 += 1
                if(('LN2' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc2 += 1
                if(('LN3' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc3 += 1
                if(('LN4' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc4 += 1
                if(('LN5' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc5 += 1
                if(('LN6' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc6 += 1
                if(('LN7' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc7 += 1
                if(('LN8' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc8 += 1
                if(('LN9' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc9 += 1
                if(('LN10' in attrs_graph[z]) and agent2_action == 3):
                    a3c_lc10 += 1
                    
##                if(agent1_action == 0):
##                    a0c += 1
##                if(agent2_action == 0):
##                    a0c += 1
##                if(agent1_action == 1):
##                    a1c += 1
##                if(agent2_action == 1):
##                    a1c += 1
                self.agents_map[counter].reward = self.agents_map[counter].get_offdiagonal_reward(agent1_action, agent2_action)
                self.agents_map[z.agentid-1].reward = self.agents_map[z.agentid-1].get_offdiagonal_reward(agent1_action, agent2_action)
                self.agents_map[counter].learn_agent(self.agents_map[counter].reward, agent1_action)
                self.agents_map[z.agentid-1].learn_agent(self.agents_map[z.agentid-1].reward, agent2_action)
                reward_total_in_the_iteration += (self.agents_map[counter].reward + self.agents_map[z.agentid-1].reward)
                self.cumulative_reward_sum += (self.agents_map[counter].reward + self.agents_map[z.agentid-1].reward)
                self.total_reward[iteration] += (self.agents_map[counter].reward + self.agents_map[z.agentid-1].reward)
                counter += 1
            # at the end of each iteration perform:
            self.cumulative_reward_sum_array.append(self.cumulative_reward_sum)
            self.average_reward[iteration] = (reward_total_in_the_iteration / len(self.agents_map))

            #print("norm1_count_lc1 - population_size", norm1_count_lc1, " - ", population_size)
            #print("norm1_count_lc2 - population_size", norm1_count_lc2, " - ", population_size)

            #print("self.norm1_frequency_array_lc1", self.norm1_frequency_array_lc1)

            self.norm1_frequency_array_lc1.append(norm1_count_lc1/population_size)
            self.norm1_frequency_array_lc2.append(norm1_count_lc2/population_size)
            self.norm1_frequency_array_lc3.append(norm1_count_lc3/population_size)
            self.norm1_frequency_array_lc4.append(norm1_count_lc4/population_size)
            self.norm1_frequency_array_lc5.append(norm1_count_lc5/population_size)
            self.norm1_frequency_array_lc6.append(norm1_count_lc6/population_size)
            self.norm1_frequency_array_lc7.append(norm1_count_lc7/population_size)
            self.norm1_frequency_array_lc8.append(norm1_count_lc8/population_size)
            self.norm1_frequency_array_lc9.append(norm1_count_lc9/population_size)
            self.norm1_frequency_array_lc10.append(norm1_count_lc10/population_size)

            self.norm2_frequency_array_lc1.append(norm2_count_lc1/population_size)
            self.norm2_frequency_array_lc2.append(norm2_count_lc2/population_size)
            self.norm2_frequency_array_lc3.append(norm2_count_lc3/population_size)
            self.norm2_frequency_array_lc4.append(norm2_count_lc4/population_size)
            self.norm2_frequency_array_lc5.append(norm2_count_lc5/population_size)
            self.norm2_frequency_array_lc6.append(norm2_count_lc6/population_size)
            self.norm2_frequency_array_lc7.append(norm2_count_lc7/population_size)
            self.norm2_frequency_array_lc8.append(norm2_count_lc8/population_size)
            self.norm2_frequency_array_lc9.append(norm2_count_lc9/population_size)
            self.norm2_frequency_array_lc10.append(norm2_count_lc10/population_size)

            self.norm3_frequency_array_lc1.append(norm3_count_lc1/population_size)
            self.norm3_frequency_array_lc2.append(norm3_count_lc2/population_size)
            self.norm3_frequency_array_lc3.append(norm3_count_lc3/population_size)
            self.norm3_frequency_array_lc4.append(norm3_count_lc4/population_size)
            self.norm3_frequency_array_lc5.append(norm3_count_lc5/population_size)
            self.norm3_frequency_array_lc6.append(norm3_count_lc6/population_size)
            self.norm3_frequency_array_lc7.append(norm3_count_lc7/population_size)
            self.norm3_frequency_array_lc8.append(norm3_count_lc8/population_size)
            self.norm3_frequency_array_lc9.append(norm3_count_lc9/population_size)
            self.norm3_frequency_array_lc10.append(norm3_count_lc10/population_size)

            self.norm4_frequency_array_lc1.append(norm4_count_lc1/population_size)
            self.norm4_frequency_array_lc2.append(norm4_count_lc2/population_size)
            self.norm4_frequency_array_lc3.append(norm4_count_lc3/population_size)
            self.norm4_frequency_array_lc4.append(norm4_count_lc4/population_size)
            self.norm4_frequency_array_lc5.append(norm4_count_lc5/population_size)
            self.norm4_frequency_array_lc6.append(norm4_count_lc6/population_size)
            self.norm4_frequency_array_lc7.append(norm4_count_lc7/population_size)
            self.norm4_frequency_array_lc8.append(norm4_count_lc8/population_size)
            self.norm4_frequency_array_lc9.append(norm4_count_lc9/population_size)
            self.norm4_frequency_array_lc10.append(norm4_count_lc10/population_size)
            
##            self.norm1_frequency_array_lc1.append(a0c_lc1/population_size)
##            self.norm1_frequency_array_lc2.append(a0c_lc2/population_size)
##            self.norm1_frequency_array_lc3.append(a0c_lc3/population_size)
##            self.norm1_frequency_array_lc4.append(a0c_lc4/population_size)
##            self.norm1_frequency_array_lc5.append(a0c_lc5/population_size)
##            self.norm1_frequency_array_lc6.append(a0c_lc6/population_size)
##            self.norm1_frequency_array_lc7.append(a0c_lc7/population_size)
##            self.norm1_frequency_array_lc8.append(a0c_lc8/population_size)
##            self.norm1_frequency_array_lc9.append(a0c_lc9/population_size)
##            self.norm1_frequency_array_lc10.append(a0c_lc10/population_size)
##
##            self.norm2_frequency_array_lc1.append(a1c_lc1/population_size)
##            self.norm2_frequency_array_lc2.append(a1c_lc2/population_size)
##            self.norm2_frequency_array_lc3.append(a1c_lc3/population_size)
##            self.norm2_frequency_array_lc4.append(a1c_lc4/population_size)
##            self.norm2_frequency_array_lc5.append(a1c_lc5/population_size)
##            self.norm2_frequency_array_lc6.append(a1c_lc6/population_size)
##            self.norm2_frequency_array_lc7.append(a1c_lc7/population_size)
##            self.norm2_frequency_array_lc8.append(a1c_lc8/population_size)
##            self.norm2_frequency_array_lc9.append(a1c_lc9/population_size)
##            self.norm2_frequency_array_lc10.append(a1c_lc10/population_size)
##
##            self.norm3_frequency_array_lc1.append(a2c_lc1/population_size)
##            self.norm3_frequency_array_lc2.append(a2c_lc2/population_size)
##            self.norm3_frequency_array_lc3.append(a2c_lc3/population_size)
##            self.norm3_frequency_array_lc4.append(a2c_lc4/population_size)
##            self.norm3_frequency_array_lc5.append(a2c_lc5/population_size)
##            self.norm3_frequency_array_lc6.append(a2c_lc6/population_size)
##            self.norm3_frequency_array_lc7.append(a2c_lc7/population_size)
##            self.norm3_frequency_array_lc8.append(a2c_lc8/population_size)
##            self.norm3_frequency_array_lc9.append(a2c_lc9/population_size)
##            self.norm3_frequency_array_lc10.append(a2c_lc10/population_size)
##
##            self.norm4_frequency_array_lc1.append(a3c_lc1/population_size)
##            self.norm4_frequency_array_lc2.append(a3c_lc2/population_size)
##            self.norm4_frequency_array_lc3.append(a3c_lc3/population_size)
##            self.norm4_frequency_array_lc4.append(a3c_lc4/population_size)
##            self.norm4_frequency_array_lc5.append(a3c_lc5/population_size)
##            self.norm4_frequency_array_lc6.append(a3c_lc6/population_size)
##            self.norm4_frequency_array_lc7.append(a3c_lc7/population_size)
##            self.norm4_frequency_array_lc8.append(a3c_lc8/population_size)
##            self.norm4_frequency_array_lc9.append(a3c_lc9/population_size)
##            self.norm4_frequency_array_lc10.append(a3c_lc10/population_size)


            #self.norm1_frequency_array.append(norm1_count/len(self.agents_map))
            #self.norm2_frequency_array.append(norm2_count/len(self.agents_map))
            self.norms_cumulative[iteration] = norms_cumulative_count
            
            atc_lc1 = a0c_lc1 + a1c_lc1 + a2c_lc1 + a3c_lc1
            atc_lc2 = a0c_lc2 + a1c_lc2 + a2c_lc2 + a3c_lc2
            atc_lc3 = a0c_lc3 + a1c_lc3 + a2c_lc3 + a3c_lc3
            atc_lc4 = a0c_lc4 + a1c_lc4 + a2c_lc4 + a3c_lc4
            atc_lc5 = a0c_lc5 + a1c_lc5 + a2c_lc5 + a3c_lc5
            atc_lc6 = a0c_lc6 + a1c_lc6 + a2c_lc6 + a3c_lc6
            atc_lc7 = a0c_lc7 + a1c_lc7 + a2c_lc7 + a3c_lc7
            atc_lc8 = a0c_lc8 + a1c_lc8 + a2c_lc8 + a3c_lc8
            atc_lc9 = a0c_lc9 + a1c_lc9 + a2c_lc9 + a3c_lc9
            atc_lc10 = a0c_lc10 + a1c_lc10 + a2c_lc10 + a3c_lc10

            #print ("atc_lc1: ", atc_lc1)
            #print ("atc_lc2: ", atc_lc2)
            #print ("atc_lc3: ", atc_lc3)

            p0_lc1 = a0c_lc1/atc_lc1    
            p1_lc1 = a1c_lc1/atc_lc1
            p2_lc1 = a2c_lc1/atc_lc1 
            p3_lc1 = a3c_lc1/atc_lc1 
            
            p0_lc2 = a0c_lc2/atc_lc2
            p1_lc2 = a1c_lc2/atc_lc2
            p2_lc2 = a2c_lc2/atc_lc2
            p3_lc2 = a3c_lc2/atc_lc2
            
            p0_lc3 = a0c_lc3/atc_lc3
            p1_lc3 = a1c_lc3/atc_lc3
            p2_lc3 = a2c_lc3/atc_lc3
            p3_lc3 = a3c_lc3/atc_lc3

            p0_lc4 = a0c_lc4/atc_lc4    
            p1_lc4 = a1c_lc4/atc_lc4
            p2_lc4 = a2c_lc4/atc_lc4
            p3_lc4 = a3c_lc4/atc_lc4
            
            p0_lc5 = a0c_lc5/atc_lc5    
            p1_lc5 = a1c_lc5/atc_lc5
            p2_lc5 = a2c_lc5/atc_lc5
            p3_lc5 = a3c_lc5/atc_lc5
            
            p0_lc6 = a0c_lc6/atc_lc6    
            p1_lc6 = a1c_lc6/atc_lc6
            p2_lc6 = a2c_lc6/atc_lc6
            p3_lc6 = a3c_lc6/atc_lc6
            
            p0_lc7 = a0c_lc7/atc_lc7    
            p1_lc7 = a1c_lc7/atc_lc7
            p2_lc7 = a2c_lc7/atc_lc7
            p3_lc7 = a3c_lc7/atc_lc7
            
            p0_lc8 = a0c_lc8/atc_lc8    
            p1_lc8 = a1c_lc8/atc_lc8
            p2_lc8 = a2c_lc8/atc_lc8
            p3_lc8 = a3c_lc8/atc_lc8
            
            p0_lc9 = a0c_lc9/atc_lc9    
            p1_lc9 = a1c_lc9/atc_lc9
            p2_lc9 = a2c_lc9/atc_lc9
            p3_lc9 = a3c_lc9/atc_lc9
            
            p0_lc10 = a0c_lc10/atc_lc10   
            p1_lc10 = a1c_lc10/atc_lc10
            p2_lc10 = a2c_lc10/atc_lc10
            p3_lc10 = a3c_lc10/atc_lc10

            #print ("p0_lc1: ", p0_lc1)
            #print ("p1_lc1: ", p1_lc1)
            #print ("p0_lc2: ", p0_lc2)
            #print ("p1_lc2: ", p1_lc2)
            #print ("p0_lc3: ", p0_lc3)
            #print ("p1_lc3: ", p1_lc3)

            #atc = a0c + a1c
            #p0 = a0c/atc # probability of action 0
            #p1 = a1c/atc # probability of action 1

            if (p0_lc1 != 0):
                p0_lc1_log = ((p0_lc1) * (-math.log2(p0_lc1)))
            else: p0_lc1_log = 0
            if (p1_lc1 != 0):
                p1_lc1_log = ((p1_lc1) * (-math.log2(p1_lc1)))
            else: p1_lc1_log = 0
            if (p2_lc1 != 0):
                p2_lc1_log = ((p2_lc1) * (-math.log2(p2_lc1)))
            else: p2_lc1_log = 0
            if (p3_lc1 != 0):
                p3_lc1_log = ((p3_lc1) * (-math.log2(p3_lc1)))
            else: p3_lc1_log = 0
            
            if (p0_lc2 != 0):
                p0_lc2_log = ((p0_lc2) * (-math.log2(p0_lc2)))
            else: p0_lc2_log = 0
            if (p1_lc2 != 0):
                p1_lc2_log = ((p1_lc2) * (-math.log2(p1_lc2)))
            else: p1_lc2_log = 0
            if (p2_lc2 != 0):
                p2_lc2_log = ((p2_lc2) * (-math.log2(p2_lc2)))
            else: p2_lc2_log = 0
            if (p3_lc2 != 0):
                p3_lc2_log = ((p3_lc2) * (-math.log2(p3_lc2)))
            else: p3_lc2_log = 0

            
            if (p0_lc3 != 0):
                p0_lc3_log = ((p0_lc3) * (-math.log2(p0_lc3)))
            else: p0_lc3_log = 0
            if (p1_lc3 != 0):
                p1_lc3_log = ((p1_lc3) * (-math.log2(p1_lc3)))
            else: p1_lc3_log = 0
            if (p2_lc3 != 0):
                p2_lc3_log = ((p2_lc3) * (-math.log2(p2_lc3)))
            else: p2_lc3_log = 0
            if (p3_lc3 != 0):
                p3_lc3_log = ((p3_lc3) * (-math.log2(p3_lc3)))
            else: p3_lc3_log = 0

            
            if (p0_lc4 != 0):
                p0_lc4_log = ((p0_lc4) * (-math.log2(p0_lc4)))
            else: p0_lc4_log = 0
            if (p1_lc4 != 0):
                p1_lc4_log = ((p1_lc4) * (-math.log2(p1_lc4)))
            else: p1_lc4_log = 0
            if (p2_lc4 != 0):
                p2_lc4_log = ((p2_lc4) * (-math.log2(p2_lc4)))
            else: p2_lc4_log = 0
            if (p3_lc4 != 0):
                p3_lc4_log = ((p3_lc4) * (-math.log2(p3_lc4)))
            else: p3_lc4_log = 0


            if (p0_lc5 != 0):
                p0_lc5_log = ((p0_lc5) * (-math.log2(p0_lc5)))
            else: p0_lc5_log = 0
            if (p1_lc5 != 0):
                p1_lc5_log = ((p1_lc5) * (-math.log2(p1_lc5)))
            else: p1_lc5_log = 0
            if (p2_lc5 != 0):
                p2_lc5_log = ((p2_lc5) * (-math.log2(p2_lc5)))
            else: p2_lc5_log = 0
            if (p3_lc5 != 0):
                p3_lc5_log = ((p3_lc5) * (-math.log2(p3_lc5)))
            else: p3_lc5_log = 0
            
            if (p0_lc6 != 0):
                p0_lc6_log = ((p0_lc6) * (-math.log2(p0_lc6)))
            else: p0_lc6_log = 0
            if (p1_lc6 != 0):
                p1_lc6_log = ((p1_lc6) * (-math.log2(p1_lc6)))
            else: p1_lc6_log = 0
            if (p2_lc6 != 0):
                p2_lc6_log = ((p2_lc6) * (-math.log2(p2_lc6)))
            else: p2_lc6_log = 0
            if (p3_lc6 != 0):
                p3_lc6_log = ((p3_lc6) * (-math.log2(p3_lc6)))
            else: p3_lc6_log = 0

            
            if (p0_lc7 != 0):
                p0_lc7_log = ((p0_lc7) * (-math.log2(p0_lc7)))
            else: p0_lc7_log = 0
            if (p1_lc7 != 0):
                p1_lc7_log = ((p1_lc7) * (-math.log2(p1_lc7)))
            else: p1_lc7_log = 0
            if (p2_lc7 != 0):
                p2_lc7_log = ((p2_lc7) * (-math.log2(p2_lc7)))
            else: p2_lc7_log = 0
            if (p3_lc7 != 0):
                p3_lc7_log = ((p3_lc7) * (-math.log2(p3_lc7)))
            else: p3_lc7_log = 0
            
            if (p0_lc8 != 0):
                p0_lc8_log = ((p0_lc8) * (-math.log2(p0_lc8)))
            else: p0_lc8_log = 0
            if (p1_lc8 != 0):
                p1_lc8_log = ((p1_lc8) * (-math.log2(p1_lc8)))
            else: p1_lc8_log = 0
            if (p2_lc8 != 0):
                p2_lc8_log = ((p2_lc8) * (-math.log2(p2_lc8)))
            else: p2_lc8_log = 0
            if (p3_lc8 != 0):
                p3_lc8_log = ((p3_lc8) * (-math.log2(p3_lc8)))
            else: p3_lc8_log = 0
            
            if (p0_lc9 != 0):
                p0_lc9_log = ((p0_lc9) * (-math.log2(p0_lc9)))
            else: p0_lc9_log = 0
            if (p1_lc9 != 0):
                p1_lc9_log = ((p1_lc9) * (-math.log2(p1_lc9)))
            else: p1_lc9_log = 0
            if (p2_lc9 != 0):
                p2_lc9_log = ((p2_lc9) * (-math.log2(p2_lc9)))
            else: p2_lc9_log = 0
            if (p3_lc9 != 0):
                p3_lc9_log = ((p3_lc9) * (-math.log2(p3_lc9)))
            else: p3_lc9_log = 0

            
            if (p0_lc10 != 0):
                p0_lc10_log = ((p0_lc10) * (-math.log2(p0_lc10)))
            else: p0_lc10_log = 0
            if (p1_lc10 != 0):
                p1_lc10_log = ((p1_lc10) * (-math.log2(p1_lc10)))
            else: p1_lc10_log = 0
            if (p2_lc10 != 0):
                p2_lc10_log = ((p2_lc10) * (-math.log2(p2_lc10)))
            else: p2_lc10_log = 0
            if (p3_lc10 != 0):
                p3_lc10_log = ((p3_lc10) * (-math.log2(p3_lc10)))
            else: p3_lc10_log = 0

            c_lc1 = normalization_constant * (p0_lc1_log + p1_lc1_log + p2_lc1_log + p3_lc1_log)
            c_lc2 = normalization_constant * (p0_lc2_log + p1_lc2_log + p2_lc2_log + p3_lc2_log)
            c_lc3 = normalization_constant * (p0_lc3_log + p1_lc3_log + p2_lc3_log + p3_lc3_log)
            c_lc4 = normalization_constant * (p0_lc4_log + p1_lc4_log + p2_lc4_log + p3_lc4_log)
            c_lc5 = normalization_constant * (p0_lc5_log + p1_lc5_log + p2_lc5_log + p3_lc5_log)
            c_lc6 = normalization_constant * (p0_lc6_log + p1_lc6_log + p2_lc6_log + p3_lc6_log)
            c_lc7 = normalization_constant * (p0_lc7_log + p1_lc7_log + p2_lc7_log + p3_lc7_log)
            c_lc8 = normalization_constant * (p0_lc8_log + p1_lc8_log + p2_lc8_log + p3_lc8_log)
            c_lc9 = normalization_constant * (p0_lc9_log + p1_lc9_log + p2_lc9_log + p3_lc9_log)
            c_lc10 = normalization_constant * (p0_lc10_log + p1_lc10_log + p2_lc10_log + p3_lc10_log)

            conformity_in_iteration_lc1 = (1-c_lc1)
            conformity_in_iteration_lc2 = (1-c_lc2)
            conformity_in_iteration_lc3 = (1-c_lc3)
            conformity_in_iteration_lc4 = (1-c_lc4)
            conformity_in_iteration_lc5 = (1-c_lc5)
            conformity_in_iteration_lc6 = (1-c_lc6)
            conformity_in_iteration_lc7 = (1-c_lc7)
            conformity_in_iteration_lc8 = (1-c_lc8)
            conformity_in_iteration_lc9 = (1-c_lc9)
            conformity_in_iteration_lc10 = (1-c_lc10)

            # calculations for diversity - n is network
            atc = (a0c_lc1 + a0c_lc2 + a0c_lc3 + a0c_lc4 + a0c_lc5 + a0c_lc6 + a0c_lc7 + a0c_lc8 + a0c_lc9 + a0c_lc10 +
                   a1c_lc1 + a1c_lc2 + a1c_lc3 + a1c_lc4 + a1c_lc5 + a1c_lc6 + a1c_lc7 + a1c_lc8 + a1c_lc9 + a1c_lc10 +
                   a2c_lc1 + a2c_lc2 + a2c_lc3 + a2c_lc4 + a2c_lc5 + a2c_lc6 + a2c_lc7 + a2c_lc8 + a2c_lc9 + a2c_lc10 +
                   a3c_lc1 + a3c_lc2 + a3c_lc3 + a3c_lc4 + a3c_lc5 + a3c_lc6 + a3c_lc7 + a3c_lc8 + a3c_lc9 + a3c_lc10)  
            p0_n = ((a0c_lc1 + a0c_lc2 + a0c_lc3 + a0c_lc4 + a0c_lc5 + a0c_lc6 + a0c_lc7 + a0c_lc8 + a0c_lc9 + a0c_lc10)/atc) 
            p1_n = ((a1c_lc1 + a1c_lc2 + a1c_lc3 + a1c_lc4 + a1c_lc5 + a1c_lc6 + a1c_lc7 + a1c_lc8 + a1c_lc9 + a1c_lc10)/atc) 
            p2_n = ((a2c_lc1 + a2c_lc2 + a2c_lc3 + a2c_lc4 + a2c_lc5 + a2c_lc6 + a2c_lc7 + a2c_lc8 + a2c_lc9 + a2c_lc10)/atc) 
            p3_n = ((a3c_lc1 + a3c_lc2 + a3c_lc3 + a3c_lc4 + a3c_lc5 + a3c_lc6 + a3c_lc7 + a3c_lc8 + a3c_lc9 + a3c_lc10)/atc) 

               
            #print ("p0_n: ", p0_n)
            #print ("p1_n: ", p1_n)

            #atc = a0c + a1c
            #p0 = a0c/atc # probability of action 0
            #p1 = a1c/atc # probability of action 1
            if (p0_n != 0):
                p0_n_log = ((p0_n) * (-math.log2(p0_n)))
            else: p0_n_log = 0
            if (p1_n != 0):
                p1_n_log = ((p1_n) * (-math.log2(p1_n)))
            else: p1_n_log = 0
            if (p2_n != 0):
                p2_n_log = ((p2_n) * (-math.log2(p2_n)))
            else: p2_n_log = 0
            if (p3_n != 0):
                p3_n_log = ((p3_n) * (-math.log2(p3_n)))
            else: p3_n_log = 0
            #print ("p0_n_log: ", p0_n_log)
            #print ("p1_n_log: ", p1_n_log)
            
            diversity_in_iteration = normalization_constant * (p0_n_log + p1_n_log + p2_n_log + p3_n_log)
            #print("================================================")
            #print ("diversity_in_iteration = normalization_constant * (p0log + p1log): ", diversity_in_iteration)
            #print("================================================")

            #diversity_in_iteration = normalization_constant * (  ((p0) * (-math.log2(p0))) + ((p1) * (-math.log2(p1)))  )
            self.diversity_array.append(diversity_in_iteration) 

            self.conformity_array1.append(conformity_in_iteration_lc1)
            self.conformity_array2.append(conformity_in_iteration_lc2)
            self.conformity_array3.append(conformity_in_iteration_lc3)
            self.conformity_array4.append(conformity_in_iteration_lc4)
            self.conformity_array5.append(conformity_in_iteration_lc5)
            self.conformity_array6.append(conformity_in_iteration_lc6)
            self.conformity_array7.append(conformity_in_iteration_lc7)
            self.conformity_array8.append(conformity_in_iteration_lc8)
            self.conformity_array9.append(conformity_in_iteration_lc9)
            self.conformity_array10.append(conformity_in_iteration_lc10)

        #print ("conformity_array1: ", self.conformity_array1)
        #print ("conformity_array2: ", self.conformity_array2)
        #print ("conformity_array3: ", self.conformity_array3)
            


        
class Agent:
    #idCounter = 0
    neighbors = None
    # constructor function
    def __init__(self, agentid, start_alpha, start_gamma, start_epsilon):
        self.QTable = np.zeros(4)
        #print("self.QTable in Agent constructor: ", self.QTable)
        self.agentid = agentid
        self.reward = 0
        # alpha: learning rate, often referred to as alpha or α, can simply be defined as how much you accept the new value vs the old value. 
        self.alpha = start_alpha # learning rate
        # gamma: or γ is a discount factor. It’s used to balance immediate and future reward.           
        self.gamma = start_gamma
        # epsilon: the exploration rate. Probability for exploration - set the percent you want to explore
        self.epsilon = start_epsilon
    
    def __str__(self):
        #return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])
        return str(self.agentid)

    # Off-Diagonal Reward Function to return the reward based on the action of agent 1 and agent 2
    def get_offdiagonal_reward(self, agent1action, agent2action):
        if((agent1action == 0 and agent2action == 0) or (agent1action == 1 and agent2action == 1) or (agent1action == 2 and agent2action == 2) or (agent1action == 3 and agent2action == 3)):
            return 2
        elif((agent1action == 1 and agent2action == 2) or (agent1action == 2 and agent2action == 1)):
            return 1
        else:
            return -2 
    
    # function to choose the next action
    def choose_action(self, QTable, epsilon, n_actions):
        # numpy.random.uniform - Draw samples from a uniform distribution.
        if random.uniform(0, 1) < epsilon:     
            # chooses a random action
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(QTable)
        return action

    # function for learning using SARSA learning algorithm
    def learn_agent(self, reward, action1):
        self.QTable[action1] = alpha * reward + (1-alpha) * self.QTable[action1]
        #print("self.QTable[action1]: ", self.QTable[action1])

##    def __str__(self):
##        return self.id
    
if __name__ == "__main__":
    # epsilon: the exploration rate. Probability for exploration - set the percent you want to explore
    epsilon = 0.1#0.1
    # alpha: learning rate, often referred to as alpha or α, can simply be defined as how much you accept the new value vs the old value. 
    alpha = 0.5#0.5
    # gamma: or γ is a discount factor. It’s used to balance immediate and future reward.  
    gamma = 0.85    
    # number of agents in a local community
    population_size = 50
    average_runs_conformity_array = []
    accumulated_average_degree = 0
    accumulated_separation_degree = 0
    
    for abc in range(100):
        G1Label = []
        G2Label = []
        G3Label = []
        G4Label = []
        G5Label = []
        G6Label = []
        G7Label = []
        G8Label = []
        G9Label = []
        G10Label = []

        degree = 3
        G1 = nx.random_regular_graph(degree, population_size)
        G2 = nx.random_regular_graph(degree, population_size)
        G3 = nx.random_regular_graph(degree, population_size)
        G4 = nx.random_regular_graph(degree, population_size)
        G5 = nx.random_regular_graph(degree, population_size)
        G6 = nx.random_regular_graph(degree, population_size)
        G7 = nx.random_regular_graph(degree, population_size)
        G8 = nx.random_regular_graph(degree, population_size)
        G9 = nx.random_regular_graph(degree, population_size)
        G10 = nx.random_regular_graph(degree, population_size)
        
        nx.set_node_attributes(G1, G1Label, 'LN')
        G1Label.append('LN1')
        nx.set_node_attributes(G2, G2Label, 'LN')
        G2Label.append('LN2')
        nx.set_node_attributes(G3, G3Label, 'LN')
        G3Label.append('LN3')
        nx.set_node_attributes(G4, G4Label, 'LN')
        G4Label.append('LN4')       
        nx.set_node_attributes(G5, G5Label, 'LN')
        G5Label.append('LN5')        
        nx.set_node_attributes(G6, G6Label, 'LN')
        G6Label.append('LN6')        
        nx.set_node_attributes(G7, G7Label, 'LN')
        G7Label.append('LN7')       
        nx.set_node_attributes(G8, G8Label, 'LN')
        G8Label.append('LN8')       
        nx.set_node_attributes(G9, G9Label, 'LN')
        G9Label.append('LN9')
        nx.set_node_attributes(G10, G10Label, 'LN')
        G10Label.append('LN10')
        
        graphs = (G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

        #G = nx.disjoint_union_all(graphs)

        G = nx.union_all(graphs, rename=('1','2','3','4','5','6','7','8','9','10'))
        #G = nx.convert_node_labels_to_integers(G)


        #print(G.nodes())

        #G = nx.union_all(graphs, rename=('A','B','C'))
        #nx.draw_networkx(G,with_labels=True,node_size=500)
        #plt.show()

        probability = 0.1
        number_of_edges = 50
        localcommunitycount = 11

        edgeCount = 0
        while edgeCount<number_of_edges:
            #print ("--------------------------------------")
            #print ("edgeCount: ", edgeCount)
            if random.uniform(0, 1) > probability:
                localcommunity1 = np.random.randint(1,localcommunitycount)
                #print ("localcommunity1: ", localcommunity1)
                localcommunity2 = np.random.randint(1,localcommunitycount)
                #print ("localcommunity2: ", localcommunity2)

                while (localcommunity1 == localcommunity2):
                    #print ("localcommunity1 == localcommunity2")
                    localcommunity2 = np.random.randint(1, localcommunitycount)
                    #print ("New local community2: ", localcommunity2)
                node1 = np.random.randint(0, population_size)
                #print ("node1: ", node1)
                node2 = np.random.randint(0, population_size)
                #print ("node2: ", node2)
                G.add_edge(str(localcommunity1)+str(node1), str(localcommunity2)+str(node2))
                #print ("str(localcommunity1)+str(node1),str(localcommunity2)+str(node2): ", str(localcommunity1), str(node1), str(localcommunity2), str(node2))               
            else:
                #print ("-----------------------------------------")
                #print ("Creating intra community links!!!")
                localcommunity = np.random.randint(1,localcommunitycount)
                #print ("localcommunity: ", localcommunity)
                node1 = np.random.randint(0, population_size)
                #print ("node1: ", node1)
                node2 = np.random.randint(0, population_size)
                #print ("node2: ", node2)
                while (node1 == node2):
                    #print ("node1 == node2")
                    node2 = np.random.randint(0, population_size)
                    #print ("New node2: ", node2)                    
                #print ("-----------------------------------------")
                G.add_edge(str(localcommunity)+str(node1), str(localcommunity)+str(node2))
                #print ("str(localcommunity)+str(node1), str(localcommunity)+str(node2): ", str(localcommunity), str(node1), str(node2))

            edgeCount += 1

        #nx.draw_networkx(G,with_labels=True,node_size=500)
        #plt.show()

        s = Simulation(G)
        s.simulate()

        G = s.graph
        x = nx.get_node_attributes(G, 'LN')

        LC1Nodes = []
        LC2Nodes = []
        LC3Nodes = []
        LC4Nodes = []
        LC5Nodes = []
        LC6Nodes = []
        LC7Nodes = []
        LC8Nodes = []
        LC9Nodes = []
        LC10Nodes = []

        for n in G.nodes():
            if ('LN1' in x[n]):
                LC1Nodes.append(str(n))
            if ('LN2' in x[n]):
                LC2Nodes.append(str(n))
            if ('LN3' in x[n]):
                LC3Nodes.append(str(n))
            if ('LN4' in x[n]):
                LC4Nodes.append(str(n))
            if ('LN5' in x[n]):
                LC5Nodes.append(str(n))
            if ('LN6' in x[n]):
                LC6Nodes.append(str(n))
            if ('LN7' in x[n]):
                LC7Nodes.append(str(n))
            if ('LN8' in x[n]):
                LC8Nodes.append(str(n))
            if ('LN9' in x[n]):
                LC9Nodes.append(str(n))
            if ('LN10' in x[n]):
                LC10Nodes.append(str(n))

##        p = np.arange(0, s.number_of_iterations, 1)
##        f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
##        ax1.plot(p, s.conformity_array1, label = 'Conformity - local community #1', color='green')
##        plt.ylim(ymin=0)
##        ax2.plot(p, s.conformity_array2, label = 'Conformity - local community #2', color='red')
##        plt.ylim(ymin=0)
##        ax3.plot(p, s.conformity_array3, label = 'Conformity - local community #3', color='blue')
##        plt.ylim(ymin=0)
##        ax4.plot(p, s.conformity_array4, label = 'Conformity - local community #4', color='green')
##        plt.ylim(ymin=0)
##        ax5.plot(p, s.conformity_array5, label = 'Conformity - local community #5', color='pink')
##        plt.ylim(ymin=0)
##        ax6.plot(p, s.conformity_array6, label = 'Conformity - local community #6', color='brown')
##        plt.ylim(ymin=0)
##        ax7.plot(p, s.conformity_array7, label = 'Conformity - local community #7', color='purple')
##        plt.ylim(ymin=0)
##        ax8.plot(p, s.conformity_array8, label = 'Conformity - local community #8', color='black')
##        plt.ylim(ymin=0)
##        ax9.plot(p, s.conformity_array9, label = 'Conformity - local community #9', color='silver')
##        plt.ylim(ymin=0)
##        ax10.plot(p, s.conformity_array10, label = 'Conformity - local community #10', color='yellow')
##        plt.ylim(ymin=0)
##        ax1.legend()
##        ax2.legend()
##        ax3.legend()
##        ax4.legend()
##        ax5.legend()
##        ax6.legend()
##        ax7.legend()
##        ax8.legend()
##        ax9.legend()
##        ax10.legend()
##        ax1.grid(True)
##        ax2.grid(True)
##        ax3.grid(True)
##        ax4.grid(True)
##        ax5.grid(True)
##        ax6.grid(True)
##        ax7.grid(True)
##        ax8.grid(True)
##        ax9.grid(True)
##        ax10.grid(True)
##        ax1.set_title('Plotting Individual Conformity of all Local Communities', fontsize=20)
##        ax9.set_xlabel('Iterations', fontsize=15)
##        ax4.set_ylabel('Conformity', fontsize=15)
##        f.subplots_adjust(hspace=0)
##        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
##        plt.show()
        
        # calculating average of conformity (recorded for each iteration) of all ten local communities and
        # storing in a array
        average_conformity_array = []
        i = 0
        j = len(s.conformity_array1)
        while i < j:
            average_conformity_array.append((s.conformity_array1[i] + s.conformity_array2[i] + s.conformity_array3[i] + s.conformity_array4[i] + s.conformity_array5[i] +
                                                       s.conformity_array6[i] + s.conformity_array7[i] + s.conformity_array8[i] + s.conformity_array9[i] + s.conformity_array10[i])/10)
            #average_community_conformity_array.append((s.conformity_array1[i] + s.conformity_array2[i] + s.conformity_array3[i]) /3)
            i += 1

        with open('average_conformity_ODR2_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(average_conformity_array))

        with open('conformity_lc1_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array1))
        with open('conformity_lc2_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array2))
        with open('conformity_lc3_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array3))
        with open('conformity_lc4_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array4))
        with open('conformity_lc5_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array5))
        with open('conformity_lc6_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array6))
        with open('conformity_lc7_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array7))
        with open('conformity_lc8_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array8))
        with open('conformity_lc9_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array9))
        with open('conformity_lc10_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.conformity_array10))
            
        with open('diversity_ODR2_run'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.diversity_array))

        with open('norm1_frequency_array_lc1_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc1))
        with open('norm1_frequency_array_lc2_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc2))
        with open('norm1_frequency_array_lc3_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc3))
        with open('norm1_frequency_array_lc4_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc4))
        with open('norm1_frequency_array_lc5_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc5))
        with open('norm1_frequency_array_lc6_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc6))
        with open('norm1_frequency_array_lc7_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc7))
        with open('norm1_frequency_array_lc8_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc8))
        with open('norm1_frequency_array_lc9_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc9))
        with open('norm1_frequency_array_lc10_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc10))

        with open('norm2_frequency_array_lc1_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc1))
        with open('norm2_frequency_array_lc2_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc2))
        with open('norm2_frequency_array_lc3_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc3))
        with open('norm2_frequency_array_lc4_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc4))
        with open('norm2_frequency_array_lc5_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc5))
        with open('norm2_frequency_array_lc6_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc6))
        with open('norm2_frequency_array_lc7_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc7))
        with open('norm2_frequency_array_lc8_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc8))
        with open('norm2_frequency_array_lc9_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc9))
        with open('norm2_frequency_array_lc10_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc10))

        with open('norm3_frequency_array_lc1_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc1))
        with open('norm3_frequency_array_lc2_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc2))
        with open('norm3_frequency_array_lc3_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc3))
        with open('norm3_frequency_array_lc4_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc4))
        with open('norm3_frequency_array_lc5_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc5))
        with open('norm3_frequency_array_lc6_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc6))
        with open('norm3_frequency_array_lc7_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc7))
        with open('norm3_frequency_array_lc8_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc8))
        with open('norm3_frequency_array_lc9_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc9))
        with open('norm3_frequency_array_lc10_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm3_frequency_array_lc10))

        with open('norm4_frequency_array_lc1_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc1))
        with open('norm4_frequency_array_lc2_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc2))
        with open('norm4_frequency_array_lc3_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc3))
        with open('norm4_frequency_array_lc4_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc4))
        with open('norm4_frequency_array_lc5_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc5))
        with open('norm4_frequency_array_lc6_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc6))
        with open('norm4_frequency_array_lc7_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc7))
        with open('norm4_frequency_array_lc8_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc8))
        with open('norm4_frequency_array_lc9_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc9))
        with open('norm4_frequency_array_lc10_ODR2_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm4_frequency_array_lc10))

        #print ("************************************************")
        #print ("************************************************")
        #print ("average_community_conformity_array: ", average_community_conformity_array)

    #print (average_conformity_array)

##        nx.draw_networkx(G1,with_labels=True,node_size=500)
##        plt.show()
##        
##        nx.draw_networkx(G,with_labels=True,node_size=500)
##        plt.show()
##
##        graphs1 = (G1, G2, G3)

        #G1 = nx.union_all(graphs1)#, rename=('1','2','3'))
        #G = nx.union_all(graphs, rename=('A','B','C'))
        #nx.draw_networkx(G1,with_labels=True,node_size=500)
        #plt.show()

        # calculating average degree (AD) in network G
        # AD = total number of neighbors in the network / number of nodes in the network
        # Average degree = Sum the number of neighbors of all nodes in the network and divide by number of nodes
        total_number_of_neighbors = 0
        average_degree = 0
        
        for node in G.nodes():
            neighbors = nx.all_neighbors(G, node)
            total_number_of_neighbors = len(list(neighbors)) + total_number_of_neighbors
        #print ("total_number_of_neighbors: ", total_number_of_neighbors)
        #print ("number of nodes in G: ", G.number_of_nodes())
        average_degree = total_number_of_neighbors / G.number_of_nodes()
        accumulated_average_degree = accumulated_average_degree + average_degree
        #print ("average degree in G: ", average_degree)    
        
        total_network_separation_degree = 0
        node_separation_degree = 0
        
        #print ("LC1 nodes are: ", LC1Nodes)
        #print ("LC2 nodes are: ", LC2Nodes)
        
        #print ("-------------------------------------")
        #print ("Iterating nodes of Graph G")

        # take one node at a time from G 
        for node in G.nodes():
            number_of_intra_community_neighbors = 0
            #print ("-------------------------")
            #print ("node: ", node)

            # get the neighbors of the particular node in iteration
            neighbors = nx.all_neighbors(G, node)
            number_of_neighbors = len(list(neighbors))
            #print ("Number of neighbors of node ", node, " is: ", number_of_neighbors)
            # iterate through all the neighbors of the particular node and check
            # whether they are intra-community neighbor
            for node_neighbor in nx.all_neighbors(G, node):
                #print ("node_neighbor and node: ", node_neighbor, " : ", node)
                               
                if ( ((str(node) in LC1Nodes) and (str(node_neighbor) in LC1Nodes)) or ((str(node) in LC2Nodes) and (str(node_neighbor) in LC2Nodes)) or
                    ((str(node) in LC3Nodes) and (str(node_neighbor) in LC3Nodes)) or ((str(node) in LC4Nodes) and (str(node_neighbor) in LC4Nodes)) or
                    ((str(node) in LC5Nodes) and (str(node_neighbor) in LC5Nodes)) or ((str(node) in LC6Nodes) and (str(node_neighbor) in LC6Nodes)) or
                    ((str(node) in LC7Nodes) and (str(node_neighbor) in LC7Nodes)) or ((str(node) in LC8Nodes) and (str(node_neighbor) in LC8Nodes)) or
                    ((str(node) in LC9Nodes) and (str(node_neighbor) in LC9Nodes)) or ((str(node) in LC10Nodes) and (str(node_neighbor) in LC10Nodes)) ):
                    #print ("number_of_intra_community_neighbors += 1")
                    number_of_intra_community_neighbors += 1
               
            if (number_of_intra_community_neighbors == 0 or number_of_neighbors == 0):
                node_separation_degree = 0
            else:
                node_separation_degree = number_of_intra_community_neighbors / number_of_neighbors
                #print ("Separation degree for node ", node, " is: ", node_separation_degree)
            total_network_separation_degree += node_separation_degree
        
        average_separation_degree =  total_network_separation_degree / G.number_of_nodes()
        accumulated_separation_degree = accumulated_separation_degree + average_separation_degree
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
     
    # process results array in calculate average convergence and store in an array
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("average_conformity_ODR2_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    #print ("Average conformity array of all runs: ", a_mean.tolist())
    # dump the average_runs_conformity_array in json file
    with open('average_runs_conformity_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #print ("average_runs_conformity_array: ", average_runs_conformity_array)

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc1_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc1_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc2_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc2_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc3_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc3_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc4_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc4_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc5_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc5_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc6_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc6_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc7_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc7_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc8_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc8_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc9_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc9_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------------------------------------------------
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("conformity_lc10_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_conformity_lc10_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("diversity_ODR2_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    
    with open('average_runs_diversity_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))

# ------------------------------
# ----------------------------------------------------------------------------------------------

    #-----------------
    results_norm1_frequency_lc1 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc1_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc1.append(res)
                                    
    a = np.array(results_norm1_frequency_lc1)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc1_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc2 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc2_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc2.append(res)
                                    
    a = np.array(results_norm1_frequency_lc2)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc2_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc3 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc3_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc3.append(res)
                                    
    a = np.array(results_norm1_frequency_lc3)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc3_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc4 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc4_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc4.append(res)
                                    
    a = np.array(results_norm1_frequency_lc4)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc4_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc5 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc5_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc5.append(res)
                                    
    a = np.array(results_norm1_frequency_lc5)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc5_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc6 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc6_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc6.append(res)
                                    
    a = np.array(results_norm1_frequency_lc6)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc6_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc7 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc7_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc7.append(res)
                                    
    a = np.array(results_norm1_frequency_lc7)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc7_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc8 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc8_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc8.append(res)
                                    
    a = np.array(results_norm1_frequency_lc8)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc8_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc9 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc9_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc9.append(res)
                                    
    a = np.array(results_norm1_frequency_lc9)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc9_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc10 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc10_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc10.append(res)
                                    
    a = np.array(results_norm1_frequency_lc10)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc10_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------

    results_norm2_frequency_lc1 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc1_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc1.append(res)
                                    
    a = np.array(results_norm2_frequency_lc1)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc1_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc2 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc2_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc2.append(res)
                                    
    a = np.array(results_norm2_frequency_lc2)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc2_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc3 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc3_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc3.append(res)
                                    
    a = np.array(results_norm2_frequency_lc3)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc3_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc4 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc4_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc4.append(res)
                                    
    a = np.array(results_norm2_frequency_lc4)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc4_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc5 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc5_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc5.append(res)
                                    
    a = np.array(results_norm2_frequency_lc5)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc5_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc6 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc6_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc6.append(res)
                                    
    a = np.array(results_norm2_frequency_lc6)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc6_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc7 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc7_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc7.append(res)
                                    
    a = np.array(results_norm2_frequency_lc7)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc7_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc8 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc8_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc8.append(res)
                                    
    a = np.array(results_norm2_frequency_lc8)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc8_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc9 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc9_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc9.append(res)
                                    
    a = np.array(results_norm2_frequency_lc9)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc9_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc10 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc10_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc10.append(res)
                                    
    a = np.array(results_norm2_frequency_lc10)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc10_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))

    # NORM 3
    results_norm3_frequency_lc1 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc1_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc1.append(res)
                                    
    a = np.array(results_norm3_frequency_lc1)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc1_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc2 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc2_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc2.append(res)
                                    
    a = np.array(results_norm3_frequency_lc2)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc2_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc3 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc3_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc3.append(res)
                                    
    a = np.array(results_norm3_frequency_lc3)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc3_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc4 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc4_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc4.append(res)
                                    
    a = np.array(results_norm3_frequency_lc4)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc4_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc5 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc5_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc5.append(res)
                                    
    a = np.array(results_norm3_frequency_lc5)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc5_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc6 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc6_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc6.append(res)
                                    
    a = np.array(results_norm3_frequency_lc6)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc6_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc7 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc7_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc7.append(res)
                                    
    a = np.array(results_norm3_frequency_lc7)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc7_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc8 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc8_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc8.append(res)
                                    
    a = np.array(results_norm3_frequency_lc8)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc8_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc9 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc9_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc9.append(res)
                                    
    a = np.array(results_norm3_frequency_lc9)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc9_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm3_frequency_lc10 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm3_frequency_array_lc10_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm3_frequency_lc10.append(res)
                                    
    a = np.array(results_norm3_frequency_lc10)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms3_frequency_lc10_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))

    # NORM 4

    results_norm4_frequency_lc1 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc1_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc1.append(res)
                                    
    a = np.array(results_norm4_frequency_lc1)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc1_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc2 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc2_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc2.append(res)
                                    
    a = np.array(results_norm4_frequency_lc2)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc2_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc3 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc3_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc3.append(res)
                                    
    a = np.array(results_norm4_frequency_lc3)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc3_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc4 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc4_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc4.append(res)
                                    
    a = np.array(results_norm4_frequency_lc4)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc4_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc5 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc5_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc5.append(res)
                                    
    a = np.array(results_norm4_frequency_lc5)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc5_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc6 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc6_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc6.append(res)
                                    
    a = np.array(results_norm4_frequency_lc6)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc6_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc7 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc7_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc7.append(res)
                                    
    a = np.array(results_norm4_frequency_lc7)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc7_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc8 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc8_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc8.append(res)
                                    
    a = np.array(results_norm4_frequency_lc8)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc8_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc9 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc9_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc9.append(res)
                                    
    a = np.array(results_norm4_frequency_lc9)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc9_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm4_frequency_lc10 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm4_frequency_array_lc10_ODR2"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm4_frequency_lc10.append(res)
                                    
    a = np.array(results_norm4_frequency_lc10)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms4_frequency_lc10_ODR2'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))


    print("ODR2 - Average Degree of Network = accumulated_average_degree/number of runs: ", accumulated_average_degree/100)
    print("ODR2 - Separation Degree of Network = accumulated_separation_degree/number of runs: ", accumulated_separation_degree/100)
