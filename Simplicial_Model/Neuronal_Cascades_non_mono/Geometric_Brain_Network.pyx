# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy
cimport cython
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from itertools import combinations
from scipy.spatial import distance
from math import sqrt
import gudhi

DTYPE = np.int64

ctypedef fused A:
    int
    long
ctypedef fused link_list:
    int
    long
ctypedef fused stubs:
    int
    long

ctypedef numpy.int_t DTYPE_t

@cython.boundscheck(False)

cdef class Geometric_Brain_Network:
    
    cdef public int N, GD, nGD, perturb
    cdef public str manifold, text, noise
    cdef public list nodes
    cdef public numpy.ndarray A, A_geo, A_non_geo, positions
    cdef public int time
    cdef public dict triangles
    
    def __init__(self, int size, int geometric_degree = 1, int nongeometric_degree = 0, str manifold = 'Ring', str noise_type = 'k-regular', numpy.ndarray[DTYPE_t, ndim=2] matrix = None, int perturb = 0):
        
        self.N = size  
        self.GD = geometric_degree
        self.nGD = nongeometric_degree
        self.manifold = manifold
        self.noise = noise_type
        self.perturb = perturb
        
        if self.manifold == 'lattice':
            self.A = matrix
            self.text = 'Custom network on %d nodes'%(self.N)
        else:
            self.A_geo, self.positions = self.make_geometric()
            self.text = '%s network on %d nodes'%(self.manifold, self.N)
        
            if self.nGD > 0: 
                self.A_non_geo = self.add_noise_to_geometric()
                self.text = self.text + ' with %s noise'%(self.noise)
                self.A = self.A_geo + self.A_non_geo
            else:
                self.A = self.A_geo   

        self.triangles = self.return_triangles(self.A)

    def get_neurons(self, list neurons):
        cdef Py_ssize_t i

        if len(neurons) != self.N: 
            raise ValueError('Size of the network and the number of neurons should match')
        
        self.nodes = neurons
        for i in range(self.N):
            self.nodes[i].neighborhood = list(np.nonzero(self.A[i])[0])
    
    def make_geometric(self):
        
        cdef numpy.ndarray ring_positions, random_ring_positions, distance_matrix, random_distance_matrix, A
        cdef int s
        cdef float e1,e2
        cdef Py_ssize_t i,j
        
        A = np.zeros((self.N,self.N), dtype = np.int64)
        
        ring_positions = np.zeros((2,self.N))
        random_ring_positions = np.zeros((2,self.N))

        s = int(25)

        for i in range(self.N):
            ring_positions[:,i] = (np.cos(np.pi*2*(i/self.N)),np.sin(np.pi*2*(i/self.N)))
            random_ring_positions[:,i] = (np.cos(np.pi*2*(i/self.N) + np.random.normal(0,(s*2*np.pi/self.N)**2)),np.sin(np.pi*2*(i/self.N) + np.random.normal(0,(s*2*np.pi/self.N)**2)))

        distance_matrix = distance.cdist(ring_positions.T, ring_positions.T, 'euclidean')
        random_distance_matrix = distance.cdist(random_ring_positions.T, random_ring_positions.T, 'euclidean')

        e1 = np.linalg.norm(np.array([[np.sin(np.pi*2*(1/self.N)),np.cos(np.pi*2*(1/self.N))]])-np.array([[0,1]]))*self.GD/2
        e2 = np.sort(random_distance_matrix.flatten())[len(distance_matrix[distance_matrix<=e1])-1]

        if self.manifold == 'Ring':
            
            for i in range(self.N):
                for j in range(self.N):
                    if distance_matrix[i,j]<=e1 and i!=j:
                        A[i,j] = 1
            if self.perturb > 0:
                A = self.ablate_geo_triangles(A)
            return(A, ring_positions)
            
        elif self.manifold == 'random_Ring':
            
            for i in range(self.N):
                for j in range(self.N):
                    if random_distance_matrix[i,j]<=e2 and i!=j:
                        A[i,j] = 1
            if self.perturb > 0:
                A = self.ablate_geo_triangles(A)
            return(A, random_ring_positions)
        
    def add_noise_to_geometric(self):#, numpy.ndarray[DTYPE_t, ndim=2] A):

        cdef Py_ssize_t i, m, n, k
        cdef int M, flag_1, flag_2, node_A, node_B, count, rand1, rand2, rand3, edges_build
        cdef numpy.ndarray nongeo, index, link_list, stubs, A_prime
        cdef list triangles, triangle, edge_list
        
        M = int(self.N * self.nGD)  
        
        if self.noise == 'k-regular':   ##every node has exactly nGD many nongeometric edges     
            flag_2 = True
            while flag_2:
                flag_2 = False
                #build stubs
                stubs = np.zeros((M), dtype = DTYPE)
                nongeo = np.ones((self.nGD), dtype = np.int64)
                for i in range(self.N):
                    index = (i*self.nGD) + np.arange(self.nGD, dtype = np.int64)
                    stubs[index[0]:index[-1]+1] = (i) * np.asarray(nongeo)
                    
                #build undirected link list
                link_list = np.zeros((int(M/2),2), dtype = DTYPE)
                for m in range(int(M/2)):
                    flag_1 = True # turn on flag to enter while loop
                    count = 0
                    while flag_1:
                        flag_1 = False #turn off flag to exit while loop
                        rand1 = random.randint(0, len(stubs)-1)
                        rand2 = random.randint(0, len(stubs)-1)
                    
                        node_A = int(stubs[rand1])
                        node_B = int(stubs[rand2])
                                            
                        if node_A == node_B: flag_1 = True
                    
                        for n in range(m):
                            if link_list[n,0] == node_A and link_list[n,1] == node_B:
                                flag_1 = True
                            if link_list[n,0] == node_B and link_list[n,1] == node_A:
                                flag_1 = True
                            if self.A_geo[node_A][node_B] == 1 or self.A_geo[node_B][node_A] == 1:
                                flag_1 = True
                            
                        count = count + 1
                    
                        if count > M: flag_2 = True ; break
                        
                    #make link       
                    link_list[m,0] = node_A
                    link_list[m,1] = node_B
                
                    #remove stubs from list
                    stubs = np.delete(stubs,[rand1,rand2])
        
            #build network
            A_prime = np.zeros((self.N,self.N), dtype = np.int64)
            for k in range(int(M/2)):
                A_prime[link_list[k,0],link_list[k,1]] = True
                A_prime[link_list[k,1],link_list[k,0]] = True
                
            return(A_prime)
            
        elif self.noise == 'ER-like': ## nongeometric edges are distributed so that number of expected non-geometric edges is nGD
            
            edges_build = 0
            A_prime = np.zeros((self.N,self.N), dtype = np.int64)
            while edges_build < int(M/2):
                rand1 = random.randint(0, self.N - 1)
                rand2 = random.randint(0, self.N - 1)
                if rand1 == rand2:
                    edges_build = edges_build
                elif self.A_geo[rand1, rand2] == True or self.A_geo[rand2, rand1] == True:
                    edges_build = edges_build
                else:
                    A_prime[rand1, rand2] = True
                    A_prime[rand2, rand1] = True
                    edges_build = edges_build + 1
            return(A_prime)
        
        elif self.noise == '2D_k-regular': ## adding non-geo triangles uniformly at random
            
            triangles = []
            edge_list = []
            for i in range(self.nGD):## number of non-geometric 2-simplicies to be added to each vertex. 
                #note every vertex will have self.nGD*2 many non-geometric 1-simplex via this method.
                flag1 = True
                stubs = np.arange(self.N, dtype = DTYPE)
                count = 0
                while flag1:#controls number of added non-geometric 2-simplicies which should be < self.N/3.
                    flag2 = True
                    while flag2:#controls added non-geometric 2-simplex to not be a geometric 2-simplex, as well as to not be previously added.
                        flag2 = False
                        rand1, rand2, rand3 = random.choices(stubs, k = 3)
                        if rand1 == rand2 or rand1 == rand3 or rand2 == rand3:
                            flag2 = True
                        if (rand1, rand2) in edge_list or (rand1, rand3) in edge_list or (rand2, rand3) in edge_list:
                            flag2 = True
                        if self.A_geo[rand1, rand2] == 1 or self.A_geo[rand2, rand1] == 1:
                            flag2 = True
                        if self.A_geo[rand1, rand3] == 1 or self.A_geo[rand3, rand1] == 1:
                            flag2 = True
                        if self.A_geo[rand2, rand3] == 1 or self.A_geo[rand3, rand2] == 1:
                            flag2 = True
                    edge_list.append((rand1, rand2))
                    edge_list.append((rand2, rand1))
                    edge_list.append((rand1, rand3))
                    edge_list.append((rand3, rand1))
                    edge_list.append((rand2, rand3))
                    edge_list.append((rand3, rand2))
                    triangles.append([rand1, rand2, rand3])
                    count = count + 1
                    stubs = np.delete(stubs, [np.where(stubs == rand1)])
                    stubs = np.delete(stubs, [np.where(stubs == rand2)])
                    stubs = np.delete(stubs, [np.where(stubs == rand3)])
                    if count == int(self.N/3):
                        flag1 = False
                #build network
                A_prime = np.zeros((self.N, self.N),  dtype = np.int64)
                for triangle in triangles:
                    A_prime[triangle[0], triangle[1]] = True
                    A_prime[triangle[1], triangle[0]] = True
                    A_prime[triangle[0], triangle[2]] = True
                    A_prime[triangle[2], triangle[0]] = True
                    A_prime[triangle[1], triangle[2]] = True
                    A_prime[triangle[2], triangle[1]] = True

            return(A_prime)
        
    def ablate_geo_triangles(self, numpy.ndarray[DTYPE_t, ndim=2] A):
        cdef list links_to_be_removed, random_tri, e
        cdef Py_ssize_t n, i
        cdef int flag1, m
        cdef dict tris
        
        tris = self.return_triangles(A)
        links_to_be_removed = []
        for n in range(self.N):
            flag1 = True
            while flag1:
                flag1 = False
                random_tri = random.choices(tris['%d'%n], k = self.perturb)#select a random 2-simplex attached to vertex n
                for i, e in enumerate(random_tri):
                    m = random.choice(e) #select a random vertex attached to the selected 2-simplex
                    if [n,m] in links_to_be_removed or [m,n] in links_to_be_removed:# if the selected edge of the triangle is previously selected, try again
                        flag1 = True
                    else:#if not, break one edge of the triangle
                        links_to_be_removed.append([n, m])

        for i, e in enumerate(links_to_be_removed):
            A[e[0], e[1]] = 0
            A[e[1], e[0]] = 0
        return(A)
            
    def get_nonunique_triangle_list(self, numpy.ndarray[DTYPE_t, ndim=2] A):
        
        cdef numpy.ndarray AAA, i_list, j_list, triangles_list, local_triangles_list
        cdef int total_number_triangles, counter, i, j, num_local_triangles
        cdef Py_ssize_t t, k
        
    
        AAA = np.dot(A,A)*A
        i_list, j_list = np.where(AAA) # list of edges that are involved in triangles
        total_number_triangles = int(np.sum(AAA))
    
        #now find list of non-unique triangles (i.e., [0,1,2] and [0,2,1] are both included)
        triangles_list = np.zeros((total_number_triangles,3),dtype=int)
        counter = 0
        for t in range(len(i_list)):
            i = i_list[t]
            j = j_list[t]
            num_local_triangles = AAA[i,j]
            local_triangles_list = np.where((A[i,:]+A[j,:])==2 )[0] # list of common neighbors, {k}
            #local_triangles_list = local_triangles_list[(local_triangles_list>i)* (local_triangles_list>j)]
            for k in local_triangles_list:
                triangles_list[counter] = [i,j,k]
                counter += 1 

        return(triangles_list)

    def get_nodes_unique_triangles(self, numpy.ndarray[DTYPE_t, ndim=2] nonunique_triangle_list, int node):
        cdef numpy.ndarray tri_flag

        tri_flag = nonunique_triangle_list[:,0]==node # make a flag for triangles using node i
        tri_flag = tri_flag * (nonunique_triangle_list[:,1]<nonunique_triangle_list[:,2])# keep onlyindices in ascending order
        
        return(nonunique_triangle_list[tri_flag,1:], tri_flag)

    def return_triangles(self, numpy.ndarray[DTYPE_t, ndim=2] Adjacency):
        cdef numpy.ndarray nonunique_triangle_list, p
        cdef dict triangles
        cdef Py_ssize_t i
        
        nonunique_triangle_list = self.get_nonunique_triangle_list(Adjacency)
        triangles = {}
        for i in range(self.N):
            triangles[str(i)] = [list(p) for p in self.get_nodes_unique_triangles(nonunique_triangle_list,i)[0]]
        return(triangles)
    
    def neighbor_input(self, int node_id, float K, float L = -100, str model_type = 'line_segment'):
        cdef int e
        cdef list f, active_hood, active_triangles, nbhood
        cdef float F, one_simplicies, two_simplicies
        cdef Py_ssize_t i,j

        nbhood = self.nodes[node_id].neighborhood
        active_hood = []
        active_triangles = []
        
        ## find the active hood
        for i,e in enumerate(nbhood):
            if self.nodes[e].state == 1:
                active_hood.append(e)
        
        ## find the active triangles
        for j,f in enumerate(self.triangles['%d'%node_id]):
            if self.nodes[f[0]].state == 1 and self.nodes[f[1]].state == 1:
                active_triangles.append(f)
                
        if model_type == 'line_segment':
            if len(nbhood) == 0 and len(self.triangles['%d'%node_id]) != 0:
                one_simplicies = 0
                two_simplicies = K*(len(active_triangles)/len(self.triangles['%d'%node_id]))
                
            elif len(self.triangles['%d'%node_id]) == 0 and len(nbhood) != 0:
                one_simplicies = (1-K)*(len(active_hood)/len(nbhood))
                two_simplicies = 0
                
            elif len(nbhood) == 0 and len(self.triangles['%d'%node_id]) == 0:
                one_simplicies = 0
                two_simplicies = 0
                
            else:
                one_simplicies = (1-K)*(len(active_hood)/len(nbhood))
                two_simplicies = K*(len(active_triangles)/len(self.triangles['%d'%node_id]))
                        
        elif model_type == 'linear_combination':
            one_simplicies = K*(len(active_hood)/len(nbhood))
            two_simplicies = L*(len(active_triangles)/len(self.triangles['%d'%node_id]))
            
        F = one_simplicies + two_simplicies - self.nodes[node_id].threshold
        
        return(F)
    
    def sigmoid(self, int node_id, int C, float K, float L = -100, str model_type = 'line_segment'):
        cdef float F, Z
        
        F = self.neighbor_input(node_id, K, L, model_type)
        if F == 0: F = -0.1
        Z = 1/(1+np.exp(-C*F))
        
        return(Z)
    
    def update_history(self, int node_id, int C, float K, float L = -100, str model_type = 'line_segment'):
        cdef float rand, Z
        cdef Py_ssize_t i, j
        
        rand = random.uniform(0,1)
        
        Z  = self.sigmoid(node_id, C, K, L, model_type)

        if rand <= Z:
                    
            for i in range(self.nodes[node_id].memory + 1):
                self.nodes[node_id].history.append(1)

            for j in range(self.nodes[node_id].rest):
                self.nodes[node_id].history.append(-1)
                
            self.nodes[node_id].history.append(0)
            
        else:
            self.nodes[node_id].history.append(0)
    
    def update_states(self):
        
        cdef list excited, ready_to_fire, rest
        cdef object node
        
        excited = []
        ready_to_fire = []
        rest = []
        
        for node in self.nodes:
            
            node.state = int(node.history[self.time])
                
            if node.state == 1:
                excited.append(node.name)
            elif node.state == 0:
                ready_to_fire.append(node.name)
            elif node.state == -1: 
                rest.append(node.name)
                
        return(excited, ready_to_fire, rest)
                
    def initial_spread(self, int seed):
        cdef Py_ssize_t i, j, k
        cdef set excited_nodes_set, all_nodes
        cdef int node1, node2
        cdef list excited_nodes_list

        all_nodes = set([k for k in range(self.N)])
        excited_nodes_list = self.nodes[seed].neighborhood 
        excited_nodes_set = set(excited_nodes_list)
        
        for node1 in excited_nodes_list:
            for i in range(self.nodes[node1].memory + 1):
                self.nodes[node1].history.append(1)
            for j in range(self.nodes[node1].rest):
                self.nodes[node1].history.append(-1)
                
            self.nodes[node1].history.append(0)
            
        for node2 in list(all_nodes.difference(excited_nodes_set)):
            self.nodes[node2].history.append(0)
            
    def refresh(self):
        cdef int tolerance
        cdef object node
        
        self.time = 0
        tolerance = 0
        
        for node in self.nodes:
            node.refresh_history()
            
        return(tolerance)
    
    def run_dynamic(self, int seed, int TIME, int C, float K, float L = -100, str model_type = 'line_segment'):
        
        cdef numpy.ndarray number_of_clusters, size_of_contagion
        cdef list excited_nodes, ready_to_fire_nodes, resting_nodes
        cdef int node, tolerance
        cdef Py_ssize_t t
        
        tolerance = self.refresh()
        number_of_clusters = np.zeros((TIME))
        size_of_contagion = np.zeros((TIME))
        self.initial_spread(seed)
        excited_nodes, ready_to_fire_nodes, resting_nodes = self.update_states()
        
        self.time = 1
        while self.time < TIME:
            size_of_contagion[int(self.time-1)] = len(excited_nodes)
            #number_of_clusters[int(self.time-1)] = int((np.diff(self.stack_histories(int(self.time)).T[int(self.time-1)])!=0).sum()/2)

            for node in ready_to_fire_nodes: 
                self.update_history(node, C, K, L, model_type)

            excited_nodes, ready_to_fire_nodes, resting_nodes = self.update_states()
            self.time = self.time + 1
            
        for t in range(TIME-1):
            number_of_clusters[t] = int((np.diff(self.stack_histories(TIME).T[t])!=0).sum()/2)
        
        return(size_of_contagion, number_of_clusters)
    
    def stack_histories(self, int TIME):
        cdef object node
        cdef list states
        cdef numpy.ndarray all_history
        cdef Py_ssize_t i
        
        for node in self.nodes:
            if len(node.history) < TIME:
                node.history = node.history + [node.history[-1] for i in range(len(node.history), TIME)]
            node.history = node.history[:TIME]
        states = [node.history for node in self.nodes]
        all_history = np.vstack(states)
        all_history = all_history.astype(float)
        return(all_history)
    
    def make_distance_matrix(self, int TIME, int C, int trials, float K, float L = -100, str model_type = 'line_segment'):
        cdef numpy.ndarray S, Q, H
        cdef Py_ssize_t seed

        S = np.zeros((self.N, TIME), dtype = np.int64)#contagion size
        Q = np.zeros((self.N, TIME), dtype = np.int64)#number of clusterrs
        H = np.zeros((self.N, TIME), dtype = np.int64)#histories
        
        for seed in range(self.N):
            S[seed,:], Q[seed,:] = self.run_dynamic(seed, TIME, C, K, L, model_type)
            H[seed,:] = self.stack_histories(TIME)
        
        return(S, Q, H)
    
    def compute_persistence(self, numpy.ndarray[double, ndim = 2] distances, int dimension = 2, int spy = False):
        
        cdef object rips_complex
        cdef list persistence, oned_holes
        cdef Py_ssize_t i
        cdef numpy.ndarray one_d_holes, persistence_life_times
        cdef float Delta_min, Delta_max
        
        rips_complex = gudhi.RipsComplex(distance_matrix = distances/np.max(distances), max_edge_length = 1)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = dimension)
        persistence = simplex_tree.persistence(min_persistence = 0.0)
        
        if spy: 
            gudhi.plot_persistence_diagram(persistence)

        oned_holes = [(0,0)]
        for i in range(len(persistence)):
            if persistence[i][0] == int(dimension-1):
                oned_holes.append(persistence[i][1])
        one_d_holes = np.array(oned_holes)
        persistence_life_times = one_d_holes[:,1]-one_d_holes[:,0]
        Delta_min = np.sort(persistence_life_times)[-1]-np.sort(persistence_life_times)[-2]
        Delta_max = np.sort(persistence_life_times)[-1]-np.sort(persistence_life_times)[1]
        return(Delta_min, Delta_max)

    def classify_time_series(self, numpy.ndarray[double, ndim = 2] time_series, float temp_av, list model):
        
        cdef float change_sum, av
        cdef Py_ssize_t i

        change_sum = 0
        for i in range(self.N):
            change_sum = change_sum + (np.diff(time_series[i,:])!=0).sum()
        av = change_sum/self.N

        if model == [0,0]:
            if av < 10:
                return(0)##no spread
            else:
                return(1)##full spread
        elif model == [0,1]:
            if 2 <= av and av < 6 and round(av, 3) != round(temp_av + 2, 3):
                return(-1)##wave
            elif round(av, 3) == round(temp_av + 2, 3) or  6 <= av:
                if self.nGD == 0:
                    return(-1)
                else:
                    return(1)##full spread, one activation, or limit circle
            else:# av < 2:
                return(0)#no spread
        elif model == [0,8]:
            if 2 <= av and av < 6 and round(av, 3) != round(temp_av + 2, 3):
                return(-1)##wave
            elif round(av, 3) == round(temp_av + 2, 3) or  6 <= av:
                return(1)##full spread, one activation, or limit circle
            else:# av < 2:
                return(0)#no spread
        return(av)
    
    def display_comm_sizes(self, list Q, list labels, int TIME, int C, int trials, int memory, int rest, float threshold, int K = -100, L = -100):
        
        cdef list argmaxs, colors
        cdef numpy.ndarray Q_mean, X
        cdef Py_ssize_t i, j
        cdef object fig, ax
        
        argmaxs = []
        colors = ['violet', 'green', 'black', 'lime', 'blue', 'orange', 'brown', 'yellow', 'red', 'turquoise', 
                  'purple']
    
        for j in range(len(Q)):
            Q_mean = np.mean(Q[j], axis = 0)
            argmaxs.append(np.argmax(Q_mean))
        
        X = np.linspace(0, int(np.min([TIME-2,np.max(argmaxs)])+1), int(np.min([TIME-2,np.max(argmaxs)])+2))
        
        fig,ax = plt.subplots(1,1, figsize = (12,8))
    
        for i in range(len(Q)):
            Q_mean = np.mean(Q[i], axis = 0)
        
            if i == 0: ax.plot(Q_mean[:int(np.min([TIME-2,np.max(argmaxs)])+2)], 
                               label = 'Threshold = %.2f'%labels[i], 
                               linestyle = 'dashed', 
                               marker = 'v',
                               color = colors[i%11])
            
            else: ax.plot(Q_mean[:int(np.min([TIME-2,np.max(argmaxs)])+2)], 
                          label = 'Threshold = %.2f'%labels[i], 
                          marker = 'v', 
                          color = colors[i%11])
            
            ax.fill_between(X, 
                            np.max(Q[i], axis = 0)[:int(np.min([TIME-2,np.max(argmaxs)])+2)], 
                            np.min(Q[i], axis = 0)[:int(np.min([TIME-2,np.max(argmaxs)])+2)], 
                            alpha = 0.2, color = colors[i%11])
            
        ax.set_title('%s, T = %d, C = %d, trials = %d,  K = %.1f, L = %.1f, Threshold = %.2f'%(self.text, TIME, C, trials, K, L, threshold), fontsize = 25)
        ax.set_xlabel('Time', fontsize = 20)
        ax.set_ylabel('Number of Active Nodes', fontsize = 20)
        ax.legend()
        return(fig,ax)
    
    
cdef class neuron(Geometric_Brain_Network):
    
    cdef public int name, state, memory, rest
    cdef public float threshold
    cdef public list history, neighborhood
    
    def __init__(self, int name, int state = False, int memory = 0, int rest = 0, float threshold = 0.1):
        self.name = name
        self.state = state
        self.memory = memory
        self.rest = rest
        self.threshold = threshold
        self.neighborhood = []

        
        self.refresh_history()
        
    def refresh_history(self):
        self.history = []