##Ten local communities with an agent population of 50 agents were created with
##two different average degrees (36 and 15) using the topology extended_barabasi_albert_graph.
##Average degree of the network was controlled using the parameters (m, p, q) “number of edges
##with which a new node attaches to existing nodes”;
##“probability value for adding an edge between existing nodes”; and “probability value of rewiring of existing edges”.
##A global community was created by union_all of the ten local community graphs.
##Conformity is calculated to each local community first and then averaged to the multi-agent system.
##The simulation was executed for 100 times and results were averaged.


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
        self.n_actions = 2
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
        #nx.draw_networkx(self.graph,with_labels=True,node_size=500)
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
        normalization_constant = 1/(math.log2(2)) # as there are two actions, cardinality is 2; for 3 it should be log2(3)


        attrs_graph = nx.get_node_attributes(self.graph, 'LN')

        for iteration in range(self.number_of_iterations):
            counter = 0
            reward_total_in_the_iteration = 0
            norm1_count = 0
            norm2_count = 0
            norms_cumulative_count = 0

            a0c_lc1 = 0 # action 0 count in each iteration to calculate conformity at each iteration
            a1c_lc1 = 0 # action 1 count in each iteration to calculate conformity at each iteration

            a0c_lc2 = 0
            a1c_lc2 = 0            

            a0c_lc3 = 0
            a1c_lc3 = 0
            
            a0c_lc4 = 0
            a1c_lc4 = 0 

            a0c_lc5 = 0
            a1c_lc5 = 0

            a0c_lc6 = 0
            a1c_lc6 = 0

            a0c_lc7 = 0
            a1c_lc7 = 0 

            a0c_lc8 = 0
            a1c_lc8 = 0

            a0c_lc9 = 0
            a1c_lc9 = 0 

            a0c_lc10 = 0
            a1c_lc10 = 0

            norm1_count_lc1 = 0
            norm2_count_lc1 = 0
            
            norm1_count_lc2 = 0
            norm2_count_lc2 = 0
            
            norm1_count_lc3 = 0
            norm2_count_lc3 = 0
            
            norm1_count_lc4 = 0
            norm2_count_lc4 = 0
            
            norm1_count_lc5 = 0
            norm2_count_lc5 = 0
            
            norm1_count_lc6 = 0
            norm2_count_lc6 = 0
            
            norm1_count_lc7 = 0
            norm2_count_lc7 = 0
            
            norm1_count_lc8 = 0
            norm2_count_lc8 = 0

            norm1_count_lc9 = 0
            norm2_count_lc9 = 0
            
            norm1_count_lc10 = 0
            norm2_count_lc10 = 0
            
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

                if((('LN1' in attrs_graph[x]) and agent1_action == 0) and (('LN1' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc1 += 1
                if((('LN1' in attrs_graph[x]) and agent1_action == 1) and (('LN1' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc1 += 1
                   
                if((('LN2' in attrs_graph[x]) and agent1_action == 0) and (('LN2' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc2 += 1
                if((('LN2' in attrs_graph[x]) and agent1_action == 1) and (('LN2' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc2 += 1
 
                if((('LN3' in attrs_graph[x]) and agent1_action == 0) and (('LN3' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc3 += 1
                if((('LN3' in attrs_graph[x]) and agent1_action == 1) and (('LN3' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc3 += 1
                   
                if((('LN4' in attrs_graph[x]) and agent1_action == 0) and (('LN4' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc4 += 1
                if((('LN4' in attrs_graph[x]) and agent1_action == 1) and (('LN4' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc4 += 1

                if((('LN5' in attrs_graph[x]) and agent1_action == 0) and (('LN5' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc5 += 1
                if((('LN5' in attrs_graph[x]) and agent1_action == 1) and (('LN5' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc5 += 1
                   
                if((('LN6' in attrs_graph[x]) and agent1_action == 0) and (('LN6' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc6 += 1
                if((('LN6' in attrs_graph[x]) and agent1_action == 1) and (('LN6' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc6 += 1

                if((('LN7' in attrs_graph[x]) and agent1_action == 0) and (('LN7' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc7 += 1
                if((('LN7' in attrs_graph[x]) and agent1_action == 1) and (('LN7' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc7 += 1
                   
                if((('LN8' in attrs_graph[x]) and agent1_action == 0) and (('LN8' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc8 += 1
                if((('LN8' in attrs_graph[x]) and agent1_action == 1) and (('LN8' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc8 += 1

                if((('LN9' in attrs_graph[x]) and agent1_action == 0) and (('LN9' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc9 += 1
                if((('LN9' in attrs_graph[x]) and agent1_action == 1) and (('LN9' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc9 += 1
                   
                if((('LN10' in attrs_graph[x]) and agent1_action == 0) and (('LN10' in attrs_graph[z]) and agent2_action == 0)):
                   norm1_count_lc10 += 1
                if((('LN10' in attrs_graph[x]) and agent1_action == 1) and (('LN10' in attrs_graph[z]) and agent2_action == 1)):
                   norm2_count_lc10 += 1

                    
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
                    
##                if(agent1_action == 0):
##                    a0c += 1
##                if(agent2_action == 0):
##                    a0c += 1
##                if(agent1_action == 1):
##                    a1c += 1
##                if(agent2_action == 1):
##                    a1c += 1
                self.agents_map[counter].reward = self.agents_map[counter].get_reward(agent1_action, agent2_action)
                self.agents_map[z.agentid-1].reward = self.agents_map[z.agentid-1].get_reward(agent1_action, agent2_action)
                self.agents_map[counter].learn_agent(self.agents_map[counter].reward, agent1_action)
                self.agents_map[z.agentid-1].learn_agent(self.agents_map[z.agentid-1].reward, agent2_action)
                reward_total_in_the_iteration += (self.agents_map[counter].reward + self.agents_map[z.agentid-1].reward)
                self.cumulative_reward_sum += (self.agents_map[counter].reward + self.agents_map[z.agentid-1].reward)
                self.total_reward[iteration] += (self.agents_map[counter].reward + self.agents_map[z.agentid-1].reward)
                counter += 1
            # at the end of each iteration perform:
            self.cumulative_reward_sum_array.append(self.cumulative_reward_sum)
            self.average_reward[iteration] = (reward_total_in_the_iteration / len(self.agents_map))
            
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

            
            self.norms_cumulative[iteration] = norms_cumulative_count
            atc_lc1 = a0c_lc1 + a1c_lc1
            atc_lc2 = a0c_lc2 + a1c_lc2
            atc_lc3 = a0c_lc3 + a1c_lc3

            atc_lc4 = a0c_lc4 + a1c_lc4
            atc_lc5 = a0c_lc5 + a1c_lc5
            atc_lc6 = a0c_lc6 + a1c_lc6
            atc_lc7 = a0c_lc7 + a1c_lc7
            atc_lc8 = a0c_lc8 + a1c_lc8
            atc_lc9 = a0c_lc9 + a1c_lc9
            atc_lc10 = a0c_lc10 + a1c_lc10

            #print ("atc_lc1: ", atc_lc1)
            #print ("atc_lc2: ", atc_lc2)
            #print ("atc_lc3: ", atc_lc3)

            p0_lc1 = a0c_lc1/atc_lc1    
            p1_lc1 = a1c_lc1/atc_lc1
            p0_lc2 = a0c_lc2/atc_lc2
            p1_lc2 = a1c_lc2/atc_lc2
            p0_lc3 = a0c_lc3/atc_lc3
            p1_lc3 = a1c_lc3/atc_lc3

            p0_lc4 = a0c_lc4/atc_lc4    
            p1_lc4 = a1c_lc4/atc_lc4
            p0_lc5 = a0c_lc5/atc_lc5    
            p1_lc5 = a1c_lc5/atc_lc5
            p0_lc6 = a0c_lc6/atc_lc6    
            p1_lc6 = a1c_lc6/atc_lc6
            p0_lc7 = a0c_lc7/atc_lc7    
            p1_lc7 = a1c_lc7/atc_lc7
            p0_lc8 = a0c_lc8/atc_lc8    
            p1_lc8 = a1c_lc8/atc_lc8
            p0_lc9 = a0c_lc9/atc_lc9    
            p1_lc9 = a1c_lc9/atc_lc9
            p0_lc10 = a0c_lc10/atc_lc10   
            p1_lc10 = a1c_lc10/atc_lc10

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

            if (p0_lc2 != 0):
                p0_lc2_log = ((p0_lc2) * (-math.log2(p0_lc2)))
            else: p0_lc2_log = 0
            if (p1_lc2 != 0):
                p1_lc2_log = ((p1_lc2) * (-math.log2(p1_lc2)))
            else: p1_lc2_log = 0

            if (p0_lc3 != 0):
                p0_lc3_log = ((p0_lc3) * (-math.log2(p0_lc3)))
            else: p0_lc3_log = 0
            if (p1_lc3 != 0):
                p1_lc3_log = ((p1_lc3) * (-math.log2(p1_lc3)))
            else: p1_lc3_log = 0

            if (p0_lc4 != 0):
                p0_lc4_log = ((p0_lc4) * (-math.log2(p0_lc4)))
            else: p0_lc4_log = 0
            if (p1_lc4 != 0):
                p1_lc4_log = ((p1_lc4) * (-math.log2(p1_lc4)))
            else: p1_lc4_log = 0

            if (p0_lc5 != 0):
                p0_lc5_log = ((p0_lc5) * (-math.log2(p0_lc5)))
            else: p0_lc5_log = 0
            if (p1_lc5 != 0):
                p1_lc5_log = ((p1_lc5) * (-math.log2(p1_lc5)))
            else: p1_lc5_log = 0

            if (p0_lc6 != 0):
                p0_lc6_log = ((p0_lc6) * (-math.log2(p0_lc6)))
            else: p0_lc6_log = 0
            if (p1_lc6 != 0):
                p1_lc6_log = ((p1_lc6) * (-math.log2(p1_lc6)))
            else: p1_lc6_log = 0

            if (p0_lc7 != 0):
                p0_lc7_log = ((p0_lc7) * (-math.log2(p0_lc7)))
            else: p0_lc7_log = 0
            if (p1_lc7 != 0):
                p1_lc7_log = ((p1_lc7) * (-math.log2(p1_lc7)))
            else: p1_lc7_log = 0

            if (p0_lc8 != 0):
                p0_lc8_log = ((p0_lc8) * (-math.log2(p0_lc8)))
            else: p0_lc8_log = 0
            if (p1_lc8 != 0):
                p1_lc8_log = ((p1_lc8) * (-math.log2(p1_lc8)))
            else: p1_lc8_log = 0

            if (p0_lc9 != 0):
                p0_lc9_log = ((p0_lc9) * (-math.log2(p0_lc9)))
            else: p0_lc9_log = 0
            if (p1_lc9 != 0):
                p1_lc9_log = ((p1_lc9) * (-math.log2(p1_lc9)))
            else: p1_lc9_log = 0

            if (p0_lc10 != 0):
                p0_lc10_log = ((p0_lc10) * (-math.log2(p0_lc10)))
            else: p0_lc10_log = 0
            if (p1_lc10 != 0):
                p1_lc10_log = ((p1_lc10) * (-math.log2(p1_lc10)))
            else: p1_lc10_log = 0

            c_lc1 = normalization_constant * (p0_lc1_log + p1_lc1_log)
            c_lc2 = normalization_constant * (p0_lc2_log + p1_lc2_log)
            c_lc3 = normalization_constant * (p0_lc3_log + p1_lc3_log)
            c_lc4 = normalization_constant * (p0_lc4_log + p1_lc4_log)
            c_lc5 = normalization_constant * (p0_lc5_log + p1_lc5_log)
            c_lc6 = normalization_constant * (p0_lc6_log + p1_lc6_log)
            c_lc7 = normalization_constant * (p0_lc7_log + p1_lc7_log)
            c_lc8 = normalization_constant * (p0_lc8_log + p1_lc8_log)
            c_lc9 = normalization_constant * (p0_lc9_log + p1_lc9_log)
            c_lc10 = normalization_constant * (p0_lc10_log + p1_lc10_log)

            conformity_in_iteration_lc1 = 1-c_lc1
            conformity_in_iteration_lc2 = 1-c_lc2
            conformity_in_iteration_lc3 = 1-c_lc3
            conformity_in_iteration_lc4 = 1-c_lc4
            conformity_in_iteration_lc5 = 1-c_lc5
            conformity_in_iteration_lc6 = 1-c_lc6
            conformity_in_iteration_lc7 = 1-c_lc7
            conformity_in_iteration_lc8 = 1-c_lc8
            conformity_in_iteration_lc9 = 1-c_lc9
            conformity_in_iteration_lc10 = 1-c_lc10
            
            # calculations for diversity - n is network
            p0_n = ((a0c_lc1 + a0c_lc2 + a0c_lc3 + a0c_lc4 + a0c_lc5 + a0c_lc6 + a0c_lc7 + a0c_lc8 + a0c_lc9 + a0c_lc10) / (a0c_lc1 + a0c_lc2 + a0c_lc3 + a0c_lc4 + a0c_lc5 + a0c_lc6 + a0c_lc7 + a0c_lc8 + a0c_lc9 + a0c_lc10 + a1c_lc1 + a1c_lc2 + a1c_lc3 + a1c_lc4 + a1c_lc5 + a1c_lc6 + a1c_lc7 + a1c_lc8 + a1c_lc9 + a1c_lc10))
            p1_n = ((a1c_lc1 + a1c_lc2 + a1c_lc3 + a1c_lc4 + a1c_lc5 + a1c_lc6 + a1c_lc7 + a1c_lc8 + a1c_lc9 + a1c_lc10) / (a0c_lc1 + a0c_lc2 + a0c_lc3 + a0c_lc4 + a0c_lc5 + a0c_lc6 + a0c_lc7 + a0c_lc8 + a0c_lc9 + a0c_lc10 + a1c_lc1 + a1c_lc2 + a1c_lc3 + a1c_lc4 + a1c_lc5 + a1c_lc6 + a1c_lc7 + a1c_lc8 + a1c_lc9 + a1c_lc10))
                    
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
            #print ("p0_n_log: ", p0_n_log)
            #print ("p1_n_log: ", p1_n_log)
            
            diversity_in_iteration = normalization_constant * (p0_n_log + p1_n_log)
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
        self.QTable = np.zeros(2)
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

    # function to return the reward based on the action of agent 1 and agent 2
    def get_reward(self, agent1action, agent2action):
        if(agent1action == agent2action):
            return 1
        else:
            return -1
    
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
    epsilon = 0.1
    # alpha: learning rate, often referred to as alpha or α, can simply be defined as how much you accept the new value vs the old value. 
    alpha = 0.5
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

        G1 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G2 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G3 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G4 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G5 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G6 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G7 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G8 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G9 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)
        G10 = nx.extended_barabasi_albert_graph(population_size, 4, 0.6, 0.3)

##        t_k = 5 
##        t_p = 0.3
##        G1 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G2 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G3 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G4 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G5 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G6 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G7 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G8 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G9 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
##        G10 = nx.connected_watts_strogatz_graph(population_size, t_k, t_p)
        
##        G1 = nx.complete_graph(50)
##        G2 = nx.complete_graph(50)
##        G3 = nx.complete_graph(50)
##        G4 = nx.complete_graph(50)
##        G5 = nx.complete_graph(50)
##        G6 = nx.complete_graph(50)
##        G7 = nx.complete_graph(50)
##        G8 = nx.complete_graph(50)
##        G9 = nx.complete_graph(50)
##        G10 = nx.complete_graph(50)

##        G1 = nx.barabasi_albert_graph(population_size,47)
##        G3 = nx.barabasi_albert_graph(population_size,47)
##        G2 = nx.barabasi_albert_graph(population_size,47)
##        G4 = nx.barabasi_albert_graph(population_size,47)
##        G5 = nx.barabasi_albert_graph(population_size,47)
##        G6 = nx.barabasi_albert_graph(population_size,47)
##        G7 = nx.barabasi_albert_graph(population_size,47)
##        G8 = nx.barabasi_albert_graph(population_size,47)
##        G9 = nx.barabasi_albert_graph(population_size,47)
##        G10 = nx.barabasi_albert_graph(population_size,47)

##        degree = 3
##        G1 = nx.random_regular_graph(degree, population_size)
##        G2 = nx.random_regular_graph(degree, population_size)
##        G3 = nx.random_regular_graph(degree, population_size)
##        G4 = nx.random_regular_graph(degree, population_size)
##        G5 = nx.random_regular_graph(degree, population_size)
##        G6 = nx.random_regular_graph(degree, population_size)
##        G7 = nx.random_regular_graph(degree, population_size)
##        G8 = nx.random_regular_graph(degree, population_size)
##        G9 = nx.random_regular_graph(degree, population_size)
##        G10 = nx.random_regular_graph(degree, population_size)

##        G1 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G2 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G3 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G4 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G5 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G6 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G7 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G8 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G9 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
##        G10 = nx.extended_barabasi_albert_graph(population_size, 42, 0.1, 0.1)
        
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
        number_of_edges = 200
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
        #print (LC1Nodes)
        #print (LC2Nodes)
        #print (LC3Nodes)

        #print ("Drawing graph after simulation") 
        #nx.draw_networkx(G,with_labels=True,node_size=500)
        #plt.show()

        #print(nx.info(G))

##        with open('diversity_array_S40_run'+str(abc)+'.json','w') as resF:
##            #print ("s.diversity_array: ", s.diversity_array)
##            resF.write(json.dumps(s.diversity_array))

        average_conformity_array = []
        i = 0
        j = len(s.conformity_array1)
        while i < j:
            average_conformity_array.append((s.conformity_array1[i] + s.conformity_array2[i] + s.conformity_array3[i] + s.conformity_array4[i] + s.conformity_array5[i] +
                                                       s.conformity_array6[i] + s.conformity_array7[i] + s.conformity_array8[i] + s.conformity_array9[i] + s.conformity_array10[i])/10)
            #average_community_conformity_array.append((s.conformity_array1[i] + s.conformity_array2[i] + s.conformity_array3[i]) /3)
            i += 1

        with open('average_conformity_S40_run'+str(abc)+'.json','w') as resF:
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
  

        with open('norm1_frequency_array_lc1_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc1))
        with open('norm1_frequency_array_lc2_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc2))
        with open('norm1_frequency_array_lc3_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc3))
        with open('norm1_frequency_array_lc4_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc4))
        with open('norm1_frequency_array_lc5_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc5))
        with open('norm1_frequency_array_lc6_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc6))
        with open('norm1_frequency_array_lc7_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc7))
        with open('norm1_frequency_array_lc8_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc8))
        with open('norm1_frequency_array_lc9_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc9))
        with open('norm1_frequency_array_lc10_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm1_frequency_array_lc10))

        with open('norm2_frequency_array_lc1_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc1))
        with open('norm2_frequency_array_lc2_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc2))
        with open('norm2_frequency_array_lc3_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc3))
        with open('norm2_frequency_array_lc4_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc4))
        with open('norm2_frequency_array_lc5_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc5))
        with open('norm2_frequency_array_lc6_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc6))
        with open('norm2_frequency_array_lc7_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc7))
        with open('norm2_frequency_array_lc8_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc8))
        with open('norm2_frequency_array_lc9_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc9))
        with open('norm2_frequency_array_lc10_S40_'+str(abc)+'.json','w') as resF:
            resF.write(json.dumps(s.norm2_frequency_array_lc10))
            
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
        
#----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
     
    # process results array in calculate average convergence and store in an array
    results = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("average_conformity_S40_run"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results.append(res)
                                    
    a = np.array(results)
    # to take the mean of each col
    a_mean = a.mean(axis=0)
    #print ("Average conformity array of all runs: ", a_mean.tolist())
    # dump the average_runs_conformity_array in json file
    with open('average_runs_conformity_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc1_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc2_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc3_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc4_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc5_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc6_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc7_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc8_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc9_S40'+'.json','w') as resF:
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
    
    with open('average_conformity_lc10_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

    #-----------------
    results_norm1_frequency_lc1 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc1_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc1.append(res)
                                    
    a = np.array(results_norm1_frequency_lc1)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc1_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc2 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc2_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc2.append(res)
                                    
    a = np.array(results_norm1_frequency_lc2)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc2_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc3 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc3_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc3.append(res)
                                    
    a = np.array(results_norm1_frequency_lc3)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc3_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc4 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc4_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc4.append(res)
                                    
    a = np.array(results_norm1_frequency_lc4)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc4_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc5 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc5_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc5.append(res)
                                    
    a = np.array(results_norm1_frequency_lc5)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc5_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc6 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc6_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc6.append(res)
                                    
    a = np.array(results_norm1_frequency_lc6)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc6_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc7 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc7_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc7.append(res)
                                    
    a = np.array(results_norm1_frequency_lc7)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc7_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc8 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc8_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc8.append(res)
                                    
    a = np.array(results_norm1_frequency_lc8)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc8_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc9 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc9_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc9.append(res)
                                    
    a = np.array(results_norm1_frequency_lc9)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc9_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm1_frequency_lc10 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm1_frequency_array_lc10_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm1_frequency_lc10.append(res)
                                    
    a = np.array(results_norm1_frequency_lc10)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms1_frequency_lc10_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------

    results_norm2_frequency_lc1 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc1_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc1.append(res)
                                    
    a = np.array(results_norm2_frequency_lc1)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc1_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc2 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc2_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc2.append(res)
                                    
    a = np.array(results_norm2_frequency_lc2)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc2_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc3 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc3_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc3.append(res)
                                    
    a = np.array(results_norm2_frequency_lc3)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc3_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc4 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc4_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc4.append(res)
                                    
    a = np.array(results_norm2_frequency_lc4)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc4_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc5 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc5_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc5.append(res)
                                    
    a = np.array(results_norm2_frequency_lc5)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc5_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc6 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc6_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc6.append(res)
                                    
    a = np.array(results_norm2_frequency_lc6)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc6_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc7 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc7_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc7.append(res)
                                    
    a = np.array(results_norm2_frequency_lc7)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc7_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc8 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc8_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc8.append(res)
                                    
    a = np.array(results_norm2_frequency_lc8)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc8_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc9 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc9_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc9.append(res)
                                    
    a = np.array(results_norm2_frequency_lc9)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc9_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))
    #-----------------
    results_norm2_frequency_lc10 = []
    #destdir = 'json/'
    for filename in listdir('.'):
            if filename.__contains__("norm2_frequency_array_lc10_S40"):
                    with open(filename,'r') as resF:
                            for line in resF:
                                    res = json.loads(line)
                                    results_norm2_frequency_lc10.append(res)
                                    
    a = np.array(results_norm2_frequency_lc10)
    a_mean = a.mean(axis=0)     # to take the mean of each col
    
    with open('average_runs_norms2_frequency_lc10_S40'+'.json','w') as resF:
   	 resF.write(json.dumps(a_mean.tolist()))


    print("S40 - Average Degree of Network = accumulated_average_degree/number of runs: ", accumulated_average_degree/100)
    print("S40 - Separation Degree of Network = accumulated_separation_degree/number of runs: ", accumulated_separation_degree/100)
