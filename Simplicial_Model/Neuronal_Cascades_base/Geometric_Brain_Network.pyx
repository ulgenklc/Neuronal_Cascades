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
    
    cdef public int N, GD, nGD
    cdef public str manifold, text, noise
    cdef public list nodes
    cdef public numpy.ndarray A, positions
    cdef public int time
    cdef public dict triangles
    
    def __init__(self, int size, int geometric_degree = 1, int nongeometric_degree = 0, str manifold = 'Ring', str noise_type = 'k-regular', numpy.ndarray[DTYPE_t, ndim=2] matrix = None):
        
        self.N = size  
        self.GD = geometric_degree
        self.nGD = nongeometric_degree
        self.manifold = manifold
        self.noise = noise_type
        
        if self.manifold == 'lattice':
            self.A = matrix
            self.text = 'Custom network on %d nodes'%(self.N)
        else:
            self.A, self.positions = self.make_geometric()
            self.text = '%s network on %d nodes'%(self.manifold, self.N)
        
            if self.nGD > 0: 
                self.add_noise_to_geometric()
                self.text = self.text + ' with %s noise'%(self.noise)
            
        self.triangles = self.return_triangles()
        
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

        s = int(self.N/10)

        for i in range(self.N):
            ring_positions[:,i] = (np.cos(np.pi*2*(i/self.N)),np.sin(np.pi*2*(i/self.N)))
            random_ring_positions[:,i] = (np.cos(np.pi*2*((i+np.random.normal(0,(s*2*np.pi/self.N)**2))/self.N)),np.sin(np.pi*2*((i+np.random.normal(0,(s*2*np.pi/self.N)**2))/self.N)))

        distance_matrix = distance.cdist(ring_positions.T, ring_positions.T, 'euclidean')
        random_distance_matrix = distance.cdist(random_ring_positions.T, random_ring_positions.T, 'euclidean')

        e1 = np.linalg.norm(np.array([[np.sin(np.pi*2*(1/self.N)),np.cos(np.pi*2*(1/self.N))]])-np.array([[0,1]]))*self.GD/2
        e2 = np.sort(random_distance_matrix.flatten())[len(distance_matrix[distance_matrix<=e1])-1]

        if self.manifold == 'Ring':
            
            for i in range(self.N):
                for j in range(self.N):
                    if distance_matrix[i,j]<=e1 and i!=j:
                        A[i,j] = 1
            return(A, ring_positions)
            
        elif self.manifold == 'random_Ring':
            
            for i in range(self.N):
                for j in range(self.N):
                    if random_distance_matrix[i,j]<=e2 and i!=j:
                        A[i,j] = 1
            return(A, random_ring_positions)
    
    def add_noise_to_geometric(self):#, numpy.ndarray[DTYPE_t, ndim=2] A):

        cdef Py_ssize_t i, m, n, k
        cdef int M, flag_2, flag_1, node_A, node_B, count, rand1, rand2, edges_build
        cdef numpy.ndarray nongeo, index
        cdef numpy.ndarray link_list
        cdef numpy.ndarray stubs 
        
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
                            if self.A[node_A][node_B] == 1 or self.A[node_B][node_A] == 1:
                                flag_1 = True
                            
                        count = count +1
                    
                        if count > M: flag_2 = True ; break
                        
                    #make link       
                    link_list[m,0] = node_A
                    link_list[m,1] = node_B
                
                    #remove stubs from list
                    stubs = np.delete(stubs,[rand1,rand2])
        
            #build network
            for k in range(int(M/2)):
                self.A[link_list[k,0],link_list[k,1]] = True
                self.A[link_list[k,1],link_list[k,0]] = True

            
        elif self.noise == 'ER-like': ## nongeometric edges are distributed so that number of expected non-geometric edges is nGD
            
            edges_build = 0
            while edges_build < int(M/2):
                rand1 = random.randint(0, self.N - 1)
                rand2 = random.randint(0, self.N - 1)
                if rand1 == rand2:
                    edges_build = edges_build
                elif self.A[rand1, rand2] == True or self.A[rand1, rand2] == True:
                    edges_build = edges_build
                else:
                    self.A[rand1, rand2] = True
                    self.A[rand2, rand1] = True
                    edges_build = edges_build + 1
        
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

        return triangles_list

    def get_nodes_unique_triangles(self, numpy.ndarray[DTYPE_t, ndim=2] nonunique_triangle_list, int i):
        cdef numpy.ndarray tri_flag
         
        tri_flag = nonunique_triangle_list[:,0]==i # make a flag for triangles using node i
        tri_flag = tri_flag * (nonunique_triangle_list[:,1]<nonunique_triangle_list[:,2])# keep only indices in ascending order

        return nonunique_triangle_list[tri_flag,1:], tri_flag

    def return_triangles(self):
        cdef numpy.ndarray nonunique_triangle_list, p
        cdef dict triangles
        cdef Py_ssize_t i
        
        nonunique_triangle_list = self.get_nonunique_triangle_list(self.A)
        triangles = {}
        for i in range(self.N):
            triangles[str(i)] = [list(p) for p in self.get_nodes_unique_triangles(nonunique_triangle_list,i)[0]]
        return triangles 
    
    def neighbor_input(self, int node_id, float K, float L = -100, str model_type = 'line_segment'):
        cdef list nbhood
        cdef Py_ssize_t i,j
        cdef int e
        cdef list f, #active_hood, active_triangles
        cdef float F, one_simplicies, two_simplicies#, economy

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
        
        #economy = two_simplicies - one_simplicies
        return(F)#, economy, active_hood, active_triangles)
    
    def sigmoid(self, int node_id, int C, float K, float L = -100, str model_type = 'line_segment'):
        
        cdef float F, Z#,economy
        #cdef list active_hood, active_triangles
        
        #F, economy, active_hood, active_triangles = self.neighbor_input(node_id, K, L, model_type)
        F = self.neighbor_input(node_id, K, L, model_type)
        if F == 0: F = -0.1
        Z = 1/(1+np.exp(-C*F))
        
        return(Z)#, economy, active_hood, active_triangles)
    
    def update_history(self, int node_id, int C, float K, float L = -100, str model_type = 'line_segment'):
        
        cdef list active_hood, active_triangles
        cdef float rand, Z, economy
        cdef Py_ssize_t i#,j
        
        rand = random.uniform(0,1)
        #Z, economy, active_hood, active_triangles = self.sigmoid(node_id, C, K, L, model_type)
        Z  = self.sigmoid(node_id, C, K, L, model_type)

        if rand <= Z:
                    
            for i in range(self.nodes[node_id].memory + 1):
                self.nodes[node_id].history.append(1)
            
            #self.nodes[node_id].economy.append((economy, active_hood, active_triangles))
            #for j in range(self.nodes[node_id].rest):
                #self.nodes[node_id].history.append(-1)
                
            #self.nodes[node_id].history.append(0)
            
        else:
            self.nodes[node_id].history.append(0)
    
    def update_states(self):
        cdef list excited, ready_to_fire #, rest
        
        cdef object node
        
        excited = []
        ready_to_fire = []
        #rest = []
        
        for node in self.nodes:
            
            node.state = int(node.history[self.time])
                
            if node.state == 1:
                excited.append(node.name)
            else: 
                ready_to_fire.append(node.name)
            #elif node.state == 0:
                #ready_to_fire.append(node.name)
            #else: rest.append(node.name)
                
        return(excited, ready_to_fire)
                
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
            #for j in range(self.nodes[node1].rest):
                #self.nodes[node1].history.append(-1)
                
            #self.nodes[node1].history.append(0)
            
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
        
        cdef numpy.ndarray activation_times, number_of_clusters
        cdef list size_of_contagion, excited_nodes, ready_to_fire_nodes#, resting_nodes
        cdef int node, flag_1, tolerance
        cdef Py_ssize_t i, t

        tolerance = self.refresh()
        activation_times = np.ones(self.N, dtype = np.int64)*TIME
        number_of_clusters = np.ones((1, TIME))
        size_of_contagion = [int(0)]
        self.initial_spread(seed)
        excited_nodes, ready_to_fire_nodes = self.update_states()
        
        self.time = 1
        
        while self.time < TIME and 0 < len(excited_nodes) and np.any(activation_times==TIME) and tolerance < 10:
            size_of_contagion.append(len(excited_nodes))
            
            activation_times[excited_nodes] = np.minimum(activation_times[excited_nodes], 
                                                         np.array([self.time]*len(excited_nodes)))
            
            for node in ready_to_fire_nodes: 
                self.update_history(node, C, K, L, model_type)
                
            flag_1 = len(excited_nodes)
            excited_nodes, ready_to_fire_nodes = self.update_states()
            
            if flag_1 == len(excited_nodes):
                tolerance = tolerance + 1
            
            self.time = self.time + 1
            
        for t in range(len(size_of_contagion)-2):
            number_of_clusters[0][t] = int((np.diff(self.stack_histories(len(size_of_contagion)).T[t])!=0).sum()/2)
        
        if len(size_of_contagion) < TIME:
            for i in range(len(size_of_contagion), TIME):
                size_of_contagion.append(size_of_contagion[-1])

        return(activation_times, np.array(size_of_contagion), number_of_clusters)
    
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
        return(all_history)
    
    def average_over_trials(self, int seed, int TIME, int C, int trials, float K, float L = -100, str model_type = 'line_segment'):
        
        cdef numpy.ndarray first_excitation_times, size_of_contagion, first_exct, contagion_size, anc_clusters, number_of_clusters
        cdef numpy.ndarray average_excitation_times, average_contagion_size, average_anc_clusters
        cdef Py_ssize_t i

        first_excitation_times = np.zeros((self.N, trials), dtype = np.int64)
        size_of_contagion = np.zeros((TIME, trials), dtype = np.int64)
        anc_clusters = np.zeros((TIME, trials), dtype = np.int64)
        
        for i in range(trials):
            first_exct, contagion_size, number_of_clusters = self.run_dynamic(seed, TIME, C, K, L, model_type)
                                                            
            first_excitation_times[:,i] = first_exct
            size_of_contagion[:,i] = contagion_size
            anc_clusters[:,i] = number_of_clusters
        
        average_excitation_times = np.mean(first_excitation_times, axis = 1)
        average_contagion_size = np.mean(size_of_contagion, axis = 1)
        average_anc_clusters = np.mean(anc_clusters, axis = 1)
        
        return(average_excitation_times, average_contagion_size, average_anc_clusters)
    
    def make_distance_matrix(self, int TIME, int C, int trials, float K, float L = -100, str model_type = 'line_segment'):
        cdef numpy.ndarray D, Q, A, distance_matrix
        cdef Py_ssize_t seed

        D = np.zeros((self.N, self.N), dtype = np.int64)
        Q = np.zeros((self.N, TIME), dtype = np.int64)
        A = np.zeros((self.N, TIME), dtype = np.int64)
        
        for seed in range(self.N):
            D[seed], Q[seed,:], A[seed,:] = self.average_over_trials(seed, TIME, C, trials, K, L, model_type)
        
        distance_matrix = euclidean_distances(D.T)
        
        return(distance_matrix, Q, A)
    
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
    cdef public list history, neighborhood#, economy
    
    def __init__(self, int name, int state = False, int memory = 0, int rest = 0, float threshold = 0.1):
        self.name = name
        self.state = state
        self.memory = memory
        self.rest = rest
        self.threshold = threshold
        self.neighborhood = []
        #self.economy = []
        
        self.refresh_history()
        
    def refresh_history(self):
        self.history = []