# Import packages
import pandas as pd
import numpy as np
import math
import time 
time1 = time.time()
# Import data file
simulated_data = pd.read_csv('Simulated Data.csv', header = 0)

# Specify model parameters
period = 1
price_lst = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]

'''
# Specify initial state: A/B/C
state = {'nA':10,'nB':10,'nC':10}

# Initialize the tree: the root node is indexed by 0, the next node is indexed by len(tree)
fashion_tree = {0:{'Week':1,'Child':[],'State':state,'Type':'D','n':0,'V':0}}


'''
# Define a function to expand a tree from a given decision node
# Add ten state-of-nature nodes (pricing options) to the tree when the focal node has no children
def Expand(tree, node):
    if len(tree[node]['Child']) == 0:
        for price in price_lst:
            tree[len(tree)] = {'Week':tree[node]['Week'],'Child':[],'Parent':node,'Price':price,'Type':'S','n':0,'V':0,'UCB':float('inf')}
        for child_node in range(10):
            tree[node]['Child'].append(len(tree)-(10-child_node))

# Define a function to perform MCTS algorithm k times from a given decision node of the tree
def BuildTree(tree, node, k):
    season = 0 
    for s in range(k):
        
        season += 1
        if season > 50:
            season = 1

        # Point to the given decision node
        pointer = node

        # Extract the state of the given decision node
        week = tree[node]['Week'] 
        nA = tree[node]['State']['nA']
        nB = tree[node]['State']['nB']
        nC = tree[node]['State']['nC']

        # Prepare a notebook to keep track of the products sold at given any initial state
        notebook = {}
        while week <= 12:
            notebook[week] = {'Revenue':0,'Delta_nA':0,'Delta_nB':0,'Delta_nC':0}
            week += 1

        # Prepare an array for product A/B/C to keep track of return values
        A_whole_return = np.array([])
        B_whole_return = np.array([])
        C_whole_return = np.array([])
        
        # Start expansion, selection and simulation, continue until reaching the end of a season or the products have been sold out
        week = tree[node]['Week']
        while week <= 12:
            # Expansion: expand if the given decision node has no child nodes
            if len(tree[pointer]['Child']) == 0:
                Expand(tree, pointer)


            # Selection: select and point to the child node with the highest UCB value, if there are multiple child nodes haven't been visited, select the first one with infinite UCB value
            Max_UCB = 0
            for n in range(10):
                if tree[tree[pointer]['Child'][n]]['UCB'] > Max_UCB:
                    Max_UCB = tree[tree[pointer]['Child'][n]]['UCB']
                    index = n
            pointer = tree[pointer]['Child'][index]


            # Simulation
            # For product A
            SKU = 'A'
            A_value = simulated_data[(simulated_data['Season'] == season) & (simulated_data['Week'] == week) & (simulated_data['SKU'] == SKU)][['ValueB1', 'ValueB2', 'ValueB3','ValueB4','ValueB5','ValueB6', 'ValueB7']]
            A_return = simulated_data[(simulated_data['Season'] == season) & (simulated_data['Week'] == week) & (simulated_data['SKU'] == SKU)][['ReturnB1', 'ReturnB2', 'ReturnB3','ReturnB4','ReturnB5','ReturnB6', 'ReturnB7']]
            A_value = np.array(A_value)
            A_return = np.array(A_return)

            # In week 12, append the return value array for the previous 11 weeks to the buyers' valuation array of week 12
            if week == 12:
                A_value = np.append(A_value, A_whole_return)

            # Check how many product A can be sold in this week
            if tree[tree[pointer]['Parent']]['State']['nA'] <= np.sum(tree[pointer]['Price'] <= A_value):
                delta_nA = tree[tree[pointer]['Parent']]['State']['nA']
            else:
                delta_nA = np.sum(tree[pointer]['Price'] <= A_value)

            # For buyers who do not buy the products in this week, record their return value
            if week <= 11:
                A_return_matched = A_return[tree[pointer]['Price'] > A_value]
                A_return_value = A_return_matched[A_return_matched != 0.0]
                A_whole_return = np.append(A_whole_return, A_return_value)

            # For product B
            SKU = 'B'

            B_value = simulated_data[(simulated_data['Season'] == season) & (simulated_data['Week'] == week) & (simulated_data['SKU'] == SKU)][['ValueB1', 'ValueB2', 'ValueB3','ValueB4','ValueB5','ValueB6', 'ValueB7']]
            B_return = simulated_data[(simulated_data['Season'] == season) & (simulated_data['Week'] == week) & (simulated_data['SKU'] == SKU)][['ReturnB1', 'ReturnB2', 'ReturnB3','ReturnB4','ReturnB5','ReturnB6', 'ReturnB7']]
            B_value = np.array(B_value)
            B_return = np.array(B_return)

            # In week 12, append return value array for the previous 11 weeks to the buyers' valuation array of week 12
            if week == 12:
                B_value = np.append(B_value, B_whole_return)

            # Check how many product B can be sold in this week
            if tree[tree[pointer]['Parent']]['State']['nB'] <= np.sum(tree[pointer]['Price'] <= B_value):
                delta_nB = tree[tree[pointer]['Parent']]['State']['nB']
            else:
                delta_nB = np.sum(tree[pointer]['Price'] <= B_value)

            # For buyers who do not buy the products in this week, record their return value
            if week <= 11:
                B_return_matched = B_return[tree[pointer]['Price'] > B_value]
                B_return_value = B_return_matched[B_return_matched != 0.0]
                B_whole_return = np.append(B_whole_return, B_return_value)

            # For product C
            SKU = 'C'

            C_value = simulated_data[(simulated_data['Season'] == season) & (simulated_data['Week'] == week) & (simulated_data['SKU'] == SKU)][['ValueB1', 'ValueB2', 'ValueB3','ValueB4','ValueB5','ValueB6', 'ValueB7']]
            C_return = simulated_data[(simulated_data['Season'] == season) & (simulated_data['Week'] == week) & (simulated_data['SKU'] == SKU)][['ReturnB1', 'ReturnB2', 'ReturnB3','ReturnB4','ReturnB5','ReturnB6', 'ReturnB7']]
            C_value = np.array(C_value)
            C_return = np.array(C_return)

            # In week 12, append return value array for the previous 11 weeks to the buyers' valuation array of week 12
            if week == 12:
                C_value = np.append(C_value, C_whole_return)

            # Check how many product C can be sold in this week
            if tree[tree[pointer]['Parent']]['State']['nC'] <= np.sum(tree[pointer]['Price'] <= C_value):
                delta_nC = tree[tree[pointer]['Parent']]['State']['nC']
            else:
                delta_nC = np.sum(tree[pointer]['Price'] <= C_value)

            if week <= 11:
                C_return_matched = C_return[tree[pointer]['Price'] > C_value]
                C_return_value = C_return_matched[C_return_matched != 0.0]
                C_whole_return = np.append(C_whole_return, C_return_value)


            # Update notebook
            revenue = tree[pointer]['Price'] * (delta_nA + delta_nB + delta_nC)
            notebook[week]['Revenue'] = revenue
            notebook[week]['Delta_nA'] = delta_nA
            notebook[week]['Delta_nB'] = delta_nB
            notebook[week]['Delta_nC'] = delta_nC

            # Now get the updated system state at the end of the week
            nA = tree[tree[pointer]['Parent']]['State']['nA'] - notebook[week]['Delta_nA']
            nB = tree[tree[pointer]['Parent']]['State']['nB'] - notebook[week]['Delta_nB']
            nC = tree[tree[pointer]['Parent']]['State']['nC'] - notebook[week]['Delta_nC']

            # Check if the products are sold out
            if (nA + nB + nC) == 0:
                break

            # Check if the state has been realized before
            n_Child = len(tree[pointer]['Child'])
            while n_Child > 0:
                if tree[tree[pointer]['Child'][n_Child - 1]]['State'] == {'nA':nA,'nB':nB,'nC':nC}:
                    pointer = tree[pointer]['Child'][n_Child - 1]
                    break
                else:
                    n_Child = n_Child - 1

            # If no, add a new decision node
            if n_Child == 0:
                tree[len(tree)] = {'Week':week+1,'Child':[],'Parent':pointer,'State':{'nA':nA,'nB':nB,'nC':nC},'Type':'D','n':0,'V':0} 
                tree[pointer]['Child'].append(len(tree)-1)
                pointer = len(tree) - 1

            # Update the current system time
            week = tree[pointer]['Week']

        # Start backpropagation (BSR is the total revenue going forward in the simulation)
        BSR = 0
        while pointer >= node:

            # Only update BSR if the node is a state-of-nature node (representing a pricing option)
            if tree[pointer]['Type'] == 'S':
                BSR = BSR + notebook[tree[pointer]['Week']]['Revenue']

            # Update V and n
            tree[pointer]['V'] = (tree[pointer]['V'] * tree[pointer]['n'] + BSR) / (tree[pointer]['n'] + 1)
            tree[pointer]['n'] = tree[pointer]['n'] + 1

            # Update UCB for the state-of-nature node
            # Update Ni and UCB for other state-of-nature nodes not chosen in the simulation
            if tree[pointer]['Type'] == 'S':
                child_lst = tree[tree[pointer]['Parent']]['Child']
                child_V_lst = []
                for child in child_lst:
                    child_V = tree[child]['V']
                    child_V_lst.append(child_V)
                if max(child_V_lst) != min(child_V_lst):   
                    adj_V = 0.9*(tree[pointer]['V']-min(child_V_lst))/(max(child_V_lst)-min(child_V_lst)) + 0.05
                    tree[pointer]['UCB'] = adj_V + 2 * (math.log(tree[tree[pointer]['Parent']]['n'] + 1) / tree[pointer]['n']) ** 0.5
                    for other in tree[tree[pointer]['Parent']]['Child']:
                        if tree[other] != tree[pointer] and tree[other]['n'] > 0:
                            other_adj_V = 0.9*(tree[other]['V']-min(child_V_lst))/(max(child_V_lst)-min(child_V_lst)) + 0.05
                            tree[other]['UCB'] = other_adj_V + 2 * (math.log(tree[tree[other]['Parent']]['n'] + 1) / tree[other]['n']) ** 0.5
                else:
                    adj_V = 0.618
                    tree[pointer]['UCB'] = adj_V + 2 * (math.log(tree[tree[pointer]['Parent']]['n'] + 1) / tree[pointer]['n']) ** 0.5
                    for other in tree[tree[pointer]['Parent']]['Child']:
                        if tree[other] != tree[pointer] and tree[other]['n'] > 0:
                            other_adj_V = 0.618
                            tree[other]['UCB'] = other_adj_V + 2 * (math.log(tree[tree[other]['Parent']]['n'] + 1) / tree[other]['n']) ** 0.5
            if pointer != node:
                pointer = tree[pointer]['Parent']
            else:
                break
            


# Define a function for optimization with the built tree
def Optimization(tree, k):
    node = 0
    while int(tree[node]['Week']) <= 12:
        print('Now is week ' + str(tree[node]['Week']) + '.')

        if tree[node]['n'] == 0:
            BuildTree(tree, node, k)

        Max_V = 0
        for n in range(10):
            if tree[tree[node]['Child'][n]]['V'] > Max_V:
                Max_V = tree[tree[node]['Child'][n]]['V']
                index = n
        node = tree[node]['Child'][index]

    
        print('For the next week, the price will be ' + str(tree[node]['Price']) + '.')
        print('During the upcoming week: ')
        delta_nA = int(input('How many product A are sold? '))
        delta_nB = int(input('How many product B are sold? '))
        delta_nC = int(input('How many product C are sold? '))

        if tree[tree[node]['Parent']]['State']['nA'] <= delta_nA:
            delta_nA = tree[tree[node]['Parent']]['State']['nA']

        if tree[tree[node]['Parent']]['State']['nB'] <= delta_nB:
            delta_nB = tree[tree[node]['Parent']]['State']['nB']

        if tree[tree[node]['Parent']]['State']['nC'] <= delta_nC:
            delta_nC = tree[tree[node]['Parent']]['State']['nC']

        nA = tree[tree[node]['Parent']]['State']['nA'] - delta_nA
        nB = tree[tree[node]['Parent']]['State']['nB'] - delta_nB
        nC = tree[tree[node]['Parent']]['State']['nC'] - delta_nC

        if nA + nB + nC == 0:
            print('Sold Out! Yeah!')
            break
        else:
            n_Child = len(tree[node]['Child'])
            while n_Child > 0:
                if tree[tree[node]['Child'][n_Child-1]]['State'] == {'nA':nA,'nB':nB, 'nC': nC}:
                    node = tree[node]['Child'][n_Child-1]
                    break
                else:
                    n_Child = n_Child - 1
            if n_Child == 0:
                tree[len(tree)] = {'Week':int(tree[node]['Week']) + 1,'Child':[],'Parent':node,'State':{'nA':nA,'nB':nB,'nC':nC},'Type':'D','n':0,'V':0} 
                tree[node]['Child'].append(len(tree) - 1)
                node = len(tree) - 1


'''
OldTreeData = pd.read_csv("tree_data.csv", low_memory = False ,header = 0, index_col = 0)
OldTree = OldTreeData.T.to_dict()

import ast
for i in range(len(OldTree)):
    OldTree[i]['Child'] = ast.literal_eval(OldTree[i]['Child'])
    if OldTree[i]['Type'] == 'D':
        OldTree[i]['State'] = ast.literal_eval(OldTree[i]['State'])
        OldTree[i].pop('Price')
        OldTree[i].pop('UCB')
    else:
        OldTree[i].pop('State')
'''
BuildTree(OldTree, 0, 40000)

# Write down the tree in a CSV file
tree_data = pd.DataFrame(data = OldTree)
transposed_tree = tree_data.T
transposed_tree.to_csv("tree_data.csv")
time2 = time.time()

print(time2-time1)
'''
#Load the tree data from a CSV file to a dictionary
OldTreeData = pd.read_csv("tree_data.csv", header = 0, index_col = 0)
OldTree = OldTreeData.T.to_dict()

import ast
for i in range(len(OldTree)):
    OldTree[i]['Child'] = ast.literal_eval(OldTree[i]['Child'])
    if OldTree[i]['Type'] == 'D':
        OldTree[i]['State'] = ast.literal_eval(OldTree[i]['State'])
        OldTree[i].pop('Price')
        OldTree[i].pop('UCB')
    else:
        OldTree[i].pop('State')
''' 
'''
# Start Optimization
Optimization(OldTree, 100)
'''