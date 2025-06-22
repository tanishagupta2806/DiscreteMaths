import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Loading data from CSV
df = pd.read_csv('data.csv')

# Creating a directed graph
G = nx.DiGraph()

# Adding edges based on the data
for index, row in df.iterrows():
    for col_name, target_email in row.items():
        if col_name != 'Email Address' and target_email != '.':
            source_email = row['Email Address']
            G.add_edge(source_email, target_email)

# plotting the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # Layout for better visualization
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=200, font_size=10, font_color='black')
plt.show()

#defining a function to perform random walk
def random_walk(graph, num_steps):
    nodes = list(graph.nodes())
    #num_nodes = len(nodes)
    node_points = {node: 0 for node in nodes}

    # Performing random walk
    for _ in range(num_steps):
        current_node = np.random.choice(nodes)  # Randomly select starting node
        for _ in range(num_steps):
            neighbors = list(graph.neighbors(current_node))
            if neighbors:
                # Randomly select a neighbor
                next_node = np.random.choice(neighbors)
                current_node = next_node
                # Increment points for the current node
                node_points[current_node] += 1

    return node_points

# Perform random walk
num_steps = 10000  
#taking the number of steps to be 10000 so that all the edges could get covered
node_points = random_walk(G, num_steps)

# Find the superleader
superleader = max(node_points, key=node_points.get)  #the node with the maximum points is the superleader
print("Superleader:", superleader)


def matrix_factorization(R, K, steps=1000, alpha=0.0002):
    
    N = len(R)
    M = len(R[0])
    #initialize the P and Q matrix with random values between -1 and 1
    P=np.random.uniform(-1,1,(N,K))
    Q=np.random.uniform(-1,1,(M,K))
    for step in range(steps):
        for i in range(N):
            for j in range(M):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[j, :].T)
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[j][k] )
                        Q[j][k] = Q[j][k] + alpha * (2 * eij * P[i][k] )

        eR = np.dot(P, Q.T)
        e = 0
        for i in range(N):
            for j in range(M):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[j, :].T), 2)
                    for k in range(K):
                        e = e + (0.002/ 2) * (pow(P[i][k], 2) + pow(Q[j][k], 2))
        if e < 0.001:
            break

    return P, Q

def find_missing_links(adjacency_matrix, K=2, steps=1000, alpha=0.0002):
    
    P, Q = matrix_factorization(adjacency_matrix, K, steps, alpha)
    reconstructed_matrix = np.dot(P, Q.T)

    missing_links = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 0 and reconstructed_matrix[i, j] > 0.75:
                missing_links.append((i, j))

    return missing_links

adjacency_matrix = nx.to_numpy_array(G)
for i in range(143):
    for j in range(143):
        if adjacency_matrix[i,j]==1 and adjacency_matrix[j,i]==0:
            adjacency_matrix[j,i]= -1

missing_links = find_missing_links(adjacency_matrix)

print("Missing Links:")
for link in missing_links:
    print(f"Node {link[0]} to Node {link[1]}")




# randomly selecting source and target nodes
nodes = list(G.nodes())
source_node = np.random.choice(nodes)
target_node = np.random.choice(nodes) 

# Calculate the shortest path
shortest_path = nx.shortest_path(G, source=source_node, target=target_node)

# Print the shortest path
print("Shortest path from", source_node, "to", target_node, ":", shortest_path)

# Calculate the length of the shortest path
shortest_path_length = nx.shortest_path_length(G, source=source_node, target=target_node)

# Print the length of the shortest path
print("Length of the shortest path:", shortest_path_length)

# Specify the source and target nodes
nodes = list(G.nodes())
source_node = np.random.choice(nodes)
target_node = np.random.choice(nodes)

# Calculate the shortest path
shortest_path = nx.shortest_path(G, source=source_node, target=target_node)

# Print the shortest path
print("Shortest path from", source_node, "to", target_node, ":", shortest_path)

# Calculate the length of the shortest path
shortest_path_length = nx.shortest_path_length(G, source=source_node, target=target_node)

# Print the length of the shortest path
print("Length of the shortest path:", shortest_path_length)
