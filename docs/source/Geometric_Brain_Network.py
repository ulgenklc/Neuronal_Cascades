#cimport numpy
#cimport cython
#import numpy as np
#import random
#import matplotlib.pyplot as plt
#from sklearn.metrics.pairwise import euclidean_distances
#from itertools import combinations
#from scipy.spatial import distance
#from math import sqrt
#import gudhi

#DTYPE = np.int64

#ctypedef fused A:
#    int
#    long
#ctypedef fused link_list:
#    int
#    long
#ctypedef fused stubs:
#   int
#    long

#ctypedef numpy.int_t DTYPE_t


class Geometric_Brain_Network():
    """
    Geometric Brain Network object to run simplicial contagions on.
        
    Attributes
    ----------
    Geometric_Brain_Network.N: int
        Size, number of nodes in the network.
    Geometric_Brain_Network.GD: int
        Geometric degree of the network.
    Geometric_Brain_Network.nGD: int
        non-Geometric degree of the network.
    Geometric_Brain_Network.manifold: str
        The geometric topology of the network. Only 'Ring' is available currently.
    Geometric_Brain_Network.text: str
        Summary of the network.
    Geometric_Brain_Network.A: array ``n x n``
        Adjacency matrix of the graph.
    Geometric_Brain_Network.nodes: List
        A list of ``neuron`` objects that corresponds to the nodes of the network in which IDs of the neurons match with the IDs of the nodes.
    Geometric_Brain_Network.time: int
        An intrinsic time property to keep track of number of iterations of the experiments.
    Geometric_Brain_Network.triangles: dict
        A dictionay of the triangles of the network where keys are node ids and values are lists of pairs of node ids that makes up a triangle together with the key value.
    
    Parameters
    -----------
    size: int
        Size of the network to be initialized.
    geometric_degree: int
        Uniform number of local neighbors that every node has.
    nongeometric_degree: int
        Fixed number of distant neighbors that every node has. 
    manifold: str
        Type of the network to be created. Only 'Ring' is available.
    """
    
    #cdef public int N, GD, nGD
    #cdef public str manifold, text
    #cdef public list nodes
    #cdef public numpy.ndarray A
    #cdef public int time
    #cdef public dict triangles
    
    def __init__(self, size, geometric_degree, nongeometric_degree, manifold): #int size, int geometric_degree = 1, int nongeometric_degree = 0, str manifold = 'Ring'):
        
        self.N = size  
        self.GD = geometric_degree
        self.nGD = nongeometric_degree
        self.manifold = manifold
        self.text = '%s Network on %d nodes'%(self.manifold, self.N)
        A = np.zeros((self.N,self.N), dtype = np.int64)
        
        self.make_geometric(A)
        
        if self.nGD > 0: self.add_noise_to_geometric()
            
        self.triangles = self.return_triangles()
        
    def get_neurons(self, neurons):#list neurons):
        
        """
        Sometimes we want to run experiments on a fixed network without changing the network connectivity. In this case, we can initialize a new set of neurons and use this method to inherit them in the network-- changing only the neuronal properties but not the connectivity.
        Parameters
        --------------
        neurons:list
            A list of ``neuron`` objects.
        
        Raises
        --------
        ValueError
            If the number of neurons and the size of the network doesn't match.
        
        """

        if len(neurons) != self.N: 
            raise ValueError('Size of the network and the number of neurons should match')
        
        self.nodes = neurons
    
    def make_geometric(self, A):#numpy.ndarray[DTYPE_t, ndim=2] A):
        """
        Method for creating a geometric ring network. This will be called upon initialization automatically.
        
        Parameters
        -----------
        A: array
            ``n x n`` matrix of zeros.
        """
        
        #cdef int gd, v
        #cdef Py_ssize_t u, i
        
        self.A = A
        
        if self.manifold == 'Ring':
            
            #if self.GD >= int(self.N)-1: 
                #raise InputError('Geometric Degree cannot exceed the half of the size of the network.')
            #elif self.GD<1 or self.GD%2 == 1:
                #raise InputError('Geometric Degree should be an even positive integer.')
            
            gd = int(self.GD/2)
            for u in range(self.N):
                for i in range(1, gd + 1):
                    #from left
                    if u + i >= self.N: 
                        v = u + i - self.N
                    else: v = u + i
                    self.A[u,v] = True
                    #from right
                    if u - i < 0: 
                        v = self.N + u - i
                    else: v = u - i
                    self.A[u,v] = True
            self.text = self.text + ' w/ GD %d'%(self.GD)
    
    def add_noise_to_geometric(self, A):#, numpy.ndarray[DTYPE_t, ndim=2] A):
        """
        This method adds non-geometric edges to the network that are long range. Every node will have ``nGD`` many
        nongeometric, long range, edges.
            
        """

        #cdef Py_ssize_t i, m, n, k
        #cdef int M, flag_2, flag_1, node_A, node_B, count, rand1, rand2
        #cdef numpy.ndarray nongeo, index
        #cdef numpy.ndarray link_list
        #cdef numpy.ndarray stubs 
        
        M = int(self.N * self.nGD)
        
        #if M%2 == 1: raise ValueError('Try providing an even non-geometric degree')
            
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
        self.text = self.text + ' and nGD %d'%self.nGD
    
    def k_cliques(self):
        """
        Fast helper method to identify triangles.
        """
        #cdef numpy.ndarray f
        #cdef int k
        #cdef list cliques
        #cdef set u,v,w, cliques_1
        
        # 2-cliques
        cliques = [set(f) for f in np.array(np.nonzero(self.A)).T]
        k = 2

        while cliques:
            # result
            yield k, cliques

            # merge k-cliques into (k+1)-cliques
            cliques_1 = set()
            for u, v in combinations(cliques, 2):
                w = u ^ v
                if len(w) == 2 and set(w) in cliques:
                    cliques_1.add(tuple(u | w))

            # remove duplicates
            cliques = list(map(set, cliques_1))
            k += 1


    def return_triangles(self,size_k =3):# int size_k = 3):
        """
        Function for getting the triangles in the network. This will be automatically called upon initialization of ``Geometric_Brain_Network``.
        
        Parameters
        ------------
        size_k: int
            By default, this is 3 since finding 3-cliques are the same as finding triangles and finding cliques is faster.
        Returns
        ---------
        triangles:dict
            A dictionay of the triangles of the network where keys are node ids and values are lists of pairs of node ids that makes up a triangle together with the key value.
            
        """
        #cdef int k,j,i
        #cdef set e, 
        #cdef list temp, clique, tris
        #cdef dict triangles
        
        for k, clique in self.k_cliques():
            if k == size_k: 
                tris = clique
        triangles = {}    
        for i in range(self.N):
            temp = []
            for j,e in enumerate(tris):
                if i in e: temp.append(list(e.difference({i})))
            triangles['%d'%i] = temp       
        return(triangles)
    
    def neighbors(self, node_id):#int node_id):
        """
        Helper function for finding the neighbors of a given node.
        
        Parameters
        --------------
        node_id: int
            ID of the node whose neighbors are going to be found.
        Returns
        ----------
        nhood:array
            Neighborhood of the given node.
        """
        
        #cdef numpy.ndarray nhood
        
        nhood = np.nonzero(self.A[node_id])[0]

        return(nhood)
    
    def neighbor_input(self, node_id, K, L, model_type = 'line_segment'):
        
        """
        
        This is a key function as it computes the current input from neighbors of a given node, R_{i}.
        
        Parameters
        ---------------
        node_id: int
            ID of the node whose input is going to be calculated.
        K: float
            Constant for weighing the edge and triangle activations.
        L: float
            Constant for weighing the edge and triangle activations. Negative unless ``model_type`` is 'linear_combination'.
        model_type: str ('line_segment' or 'linear_combination')
            We included two ways to compute the neighboring neuronal input. Former uses R_{i} = \left[(1-K)*\sum_{e \in E_{i}} \frac{e {d_{i}^{e}} + (K)*\sum_{t \in T_{i}}\frac{t}{d_{i}^{t}}\right] - \tau_{i} and the latter uses R_{i} = \left[(K)*\sum_{e \in E_{i}} \frac{e}{d_{i}^{e}} + (L)*\sum_{t \in T_{i}}\frac{t}{d_{i}^{t}}\right] - \tau_{i}. By varying K and L one can obtain different weight distributions.
        
        Returns
        ---------------
        R: float
            Neighboring neuronal input.
            
        """
        #cdef numpy.ndarray nbhood
        #cdef Py_ssize_t i,j
        #cdef int e
        #cdef list active_hood, active_triangles, f
        #cdef float F, one_simplicies, two_simplicies

        nbhood = self.neighbors(node_id)
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
            one_simplicies = (1-K)*(len(active_hood)/len(nbhood))
            two_simplicies = K*(len(active_triangles)/len(self.triangles['%d'%node_id]))
            
            R = one_simplicies + two_simplicies - self.nodes[node_id].threshold
            
        elif model_type == 'linear_combination':
            one_simplicies = K*(len(active_hood)/len(nbhood))
            two_simplicies = L*(len(active_triangles)/len(self.triangles['%d'%node_id]))
            
            R = one_simplicies + two_simplicies - self.nodes[node_id].threshold
        
        return(R)
    
    def sigmoid(self, node_id, C, K, L, model_type):# int node_id, int C, float K, float L = -100, str model_type = 'line_segment'):
        """
        Sigmoid function.
        
        Parameters
        ------------
        node_id: int
            ID of the node whose input is going to be calculated.
        C: int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        K: float
            Constant for weighing the edge and triangle activations.
        L: float
            Constant for weighing the edge and triangle activations. Negative unless ``model_type`` is 'linear_combination'.
        model_type: 'line_segment' or 'linear_combination'
            Included only to pass along ``neighbor_input``.
        
        Returns
        ------------
        Z: float
            Probability of firing.
        """
        
        #cdef float F, Z

        F = self.neighbor_input(node_id, K, L, model_type)
        Z = 1/(1+np.exp(-C*F))
        
        return(Z)
    
    def update_history(self, node_id, C, K, L, model_type):#int node_id, int C, float K, float L = -100, str model_type = 'line_segment'):
        """
        Helper method to update the history of the ``neuron`` objects at every iteration.
        
        Parameters
        ------------
        node_id: int
            ID of the node whose history is going to be updates.
        C: int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        K: float
            Constant for weighing the edge and triangle activations.
        L: float
            Constant for weighing the edge and triangle activations. Negative unless ``model_type`` is 'linear_combination'.
        model_type: 'line_segment' or 'linear_combination'
            Included only to pass along ``sigmoid``.
        """
        #cdef float rand
        #cdef Py_ssize_t i,j
        
        rand = random.uniform(0,1)
        
        if rand <= self.sigmoid(node_id, C, K, L, model_type):
            
            for i in range(self.nodes[node_id].memory + 1):
                self.nodes[node_id].history.append(1)
                
            for j in range(self.nodes[node_id].rest):
                self.nodes[node_id].history.append(-1)
                
            self.nodes[node_id].history.append(0)
            
        else:
            self.nodes[node_id].history.append(0)
    
    def update_states(self):
        """
        Helper method to update the states of ``neuron`` objects at every iteration.
        
        Returns
        ----------
        excited:list
            List of active neurons at the current ``time``.
        ready_to_fire:list
            List of neurons that are in the inactive state and ready to fire at ``time+1```.
        rest:list
            List of neurons that doesn't belong to either of those categories. This is empty as long as there are no refractory period.
        """
        #cdef list excited, ready_to_fire, rest
        #cdef object node
        
        excited = []
        ready_to_fire = []
        rest = []
        
        for node in self.nodes:
            
            node.state = int(node.history[self.time])
                
            if node.state == 1:
                excited.append(node.name)
            elif node.state == 0:
                ready_to_fire.append(node.name)
            else: rest.append(node.name)
                
        return(excited, ready_to_fire, rest)
                
    def initial_spread(self, seed):#int seed):
        """
        Helper method to activate the neighbors of the seed node with probablity 1.
        
        Parameters
        -----------
        seed:int
            Node ID of the seed node.
        """
        #cdef Py_ssize_t i, j, k
        #cdef set excited_nodes_set, all_nodes
        #cdef int node1, node2
        #cdef list excited_nodes_list

        all_nodes = set([k for k in range(self.N)])
        excited_nodes_list = list(self.neighbors(seed))
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
        """
        Helper method for setting the network time and tolerance to 0. This is necessary between different experiments for any set of parameters including ``seed``. Also, calls ``refresh_history`` which clears ``neuoron`` histories.
        Returns
        ---------
        tolerance:int
            Tolerance for experiments getting stuck at some point during contagion. Set to 0 at every trial.
        """
        #cdef int tolerance
        #cdef object node
        
        self.time = 0
        tolerance = 0
        
        for node in self.nodes:
            node.refresh_history()
            
        return(tolerance)
    
    def run_dynamic(self, seed, TIME, C, K, L, model_type):#int seed, int TIME, int C, float K, float L = -100, str model_type = 'line_segment'):
        """
        Core function that runs the experiments. There are couple control flags for computational efficiency. If ``self.time`` exceeds ``TIME``, flag. If there is no active neurons left in the network, flag. If everything gets activated once, flag. If ``tolerance`` exceeds 10, flag i.e. network repeats the exact state of itself 10 times.
        
        Parameters
        -----------
        seed:int
            Node ID of the seed node.
        TIME:int
            A limit on the number of iterations.
        C: int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        K: float
            Constant for weighing the edge and triangle activations.
        L: float
            Constant for weighing the edge and triangle activations. Negative unless ``model_type`` is 'linear_combination'.
        model_type: 'line_segment' or 'linear_combination'
            Included only to pass along ``update_states``.
        Returns
        -------------
        activation_times: array
            Activation times of all the nodes for contagions starting from seed.
        size_of_contagion: array
            Number of active nodes at every iteration.
        
        """
        
        #cdef numpy.ndarray activation_times
        #cdef list size_of_contagion, excited_nodes, ready_to_fire_nodes, resting_nodes
        #cdef int node, tolerance, flag_1
        #cdef Py_ssize_t i

        tolerance = self.refresh()
        activation_times = np.ones(self.N, dtype = np.int64)*TIME
        size_of_contagion = [int(0)]
        self.initial_spread(seed)
        excited_nodes, ready_to_fire_nodes, resting_nodes = self.update_states()
        
        self.time = 1
        
        while self.time < TIME and 0 < len(excited_nodes) and np.any(activation_times==TIME) and tolerance < 10:
            size_of_contagion.append(len(excited_nodes))
            
            activation_times[excited_nodes] = np.minimum(activation_times[excited_nodes], 
                                                         np.array([self.time]*len(excited_nodes)))
            
            
            for node in ready_to_fire_nodes: 
                self.update_history(node, C, K, L, model_type)
            
            flag_1 = len(excited_nodes)
            excited_nodes, ready_to_fire_nodes, resting_nodes = self.update_states()
            
            if flag_1 == len(excited_nodes):
                tolerance = tolerance + 1
            
            self.time = self.time + 1
        
        if len(size_of_contagion) < TIME:
            for i in range(len(size_of_contagion), TIME):
                size_of_contagion.append(size_of_contagion[-1])

        return(activation_times, np.array(size_of_contagion))
    
    def stack_histories(self,TIME):# int TIME):
        """
        Helper function for equalizing, stacking, the lengths of histories of ``neuron``s. Comes handy for visualizing single experiments.
        
        Parameters
        ------------
        TIME: int
            Number of discrete time steps for neuron histories to be visualized
        Returns
        --------
        all_history: array
            ``N x TIME`` matrix encoding histories of neurons.
        """
        #cdef object node
        #cdef list states
        #cdef numpy.ndarray all_history
        #cdef Py_ssize_t i
        
        for node in self.nodes:
            if len(node.history) < TIME:
                node.history = node.history + [node.history[-1] for i in range(len(node.history), TIME)]
            node.history = node.history[:TIME]
        states = [node.history for node in self.nodes]
        all_history = np.vstack(states)
        return(all_history)
    
    def average_over_trials(self, seed, TIME, C, trials, K, L, model_type):#int seed, int TIME, int C, int trials, float K, float L = -100, str model_type = 'line_segment'):
        """
        Helper function for averaging the activation times and contagion sizes over different trials. Trials is usually 1 unless you are doing stochastic experiments.
        
        Parameters
        -------------
        seed:int
            Node ID of the seed node.
        TIME:int
            A limit on the number of iterations.
        C: int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        trials:int
            Number of trials.
        K: float
            Constant for weighing the edge and triangle activations.
        L: float
            Constant for weighing the edge and triangle activations. Negative unless ``model_type`` is 'linear_combination'.
        model_type: 'line_segment' or 'linear_combination'
            Included only to pass along ``update_states``.
        
        Returns
        ------------
        activation_times: array
            Average activation times of all the nodes for contagions starting from seed.
        size_of_contagion: array
            Average number of active nodes at every iteration.
        """
        #cdef numpy.ndarray first_excitation_times, size_of_contagion, first_exct, contagion_size
        #cdef numpy.ndarray average_excitation_times, average_contagion_size
        #cdef Py_ssize_t i

        first_excitation_times = np.zeros((self.N, trials), dtype = np.int64)
        size_of_contagion = np.zeros((TIME, trials), dtype = np.int64)
        
        for i in range(trials):
            first_exct, contagion_size = self.run_dynamic(seed, TIME, C, K, L, model_type)
                                                            
            first_excitation_times[:,i] = first_exct
            size_of_contagion[:,i] = contagion_size
        
        average_excitation_times = np.mean(first_excitation_times, axis = 1)
        average_contagion_size = np.mean(size_of_contagion, axis = 1)
        
        return(average_excitation_times, average_contagion_size)
    
    def make_distance_matrix(self, TIME, C, trials, K, L, model_type):#int TIME, int C, int trials, float K, float L = -100, str model_type = 'line_segment'):
        """
        Main function if you are running experiments for a full set of seed nodes. This creates an activation matrix by running 
        the contagion on starting from every node and encoding the first activation times of each node. Then,
        finding the euclidean distances between the columns of this matrix, creating a distance matrix so that
        the (i,j) entry corresponds to the average time(over the trials) that a contagion reaches node j starting 
        from node i.
        
        Parameters
        ------------
        TIME:int
            A limit on the number of iterations.
        C: int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        trials:int
            Number of trials.
        K: float
            Constant for weighing the edge and triangle activations.
        L: float
            Constant for weighing the edge and triangle activations. Negative unless ``model_type`` is 'linear_combination'.
        model_type: 'line_segment' or 'linear_combination'
            Included only to pass along ``update_states``.
        
        Returns
        ---------
        distance_matrix: array
            ``n x n`` array with entries the activation times of contagions starting from node i reaching to node j.
        Q: array
            ``n x t`` array with entries number of active nodes at every time step for contagions starting at different seeds.
        
        """
        #cdef numpy.ndarray D,Q, distance_matrix
        #cdef Py_ssize_t seed

        D = np.zeros((self.N, self.N), dtype = np.int64)
        Q = np.zeros((self.N, TIME), dtype = np.int64)
        
        for seed in range(self.N):
            D[seed], Q[seed,:] = self.average_over_trials(seed, TIME, C, trials, K, L, model_type)
        
        distance_matrix = euclidean_distances(D.T)
        
        return(distance_matrix, Q)
    
    def compute_persistence(self, distances, dimension, spy):#numpy.ndarray[double, ndim = 2] distances, int dimension = 2, int spy = False):
        
        """
        Helper to compute persistent homology using the distance matrix by building a Rips filtration up to given 
        dimension(topological features to be observed are going to be one less dimensional at max).
        First normalizes the distances before the computation.
        
        Parameters
        ----------
        distances: n x n array
            distance matrix. First output of the ``make_distance_matrix``.
        dimension: int
            Max dimension of the topological features to be computed.
        spy: bool, optional
            Take a peak at the persistence diagram.
        Returns
        -------
        Delta_min: array
            Difference of the lifetimes between longest and second longest living two 1-cycles.
        Delta_max: array
            Difference of the lifetimes between longest and shortest living two 1-cycles.
        """
        
        #cdef object rips_complex
        #cdef list persistence, oned_holes
        #cdef Py_ssize_t i
        #cdef numpy.ndarray one_d_holes, persistence_life_times
        #cdef float Delta_min, Delta_max
        
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
    
    def display_comm_sizes(self,Q, labels, TIME, C, trials, threshold, K, L):# list Q, list labels, int TIME, int C, int trials, float threshold, int K = -100, L = -100):
        """
        Helper to visualize the size of the active nodes during the contagion. Shades are indicating the max 
        and min values of the spread starting from different nodes, seed node variations.
    
        Parameters
        ----------
        Q: list, [n x T+1 array]
            Output of the make_distance_matrix appended in a list
        labels: list
            Figure labels corresponding to every list element for different thresholds.
        TIME:int
            A limit on the number of iterations.
        C: int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        trials:int
            Number of trials.
        K: float
            Constant for weighing the edge and triangle activations.
        L: float
            Constant for weighing the edge and triangle activations. Negative unless ``model_type`` is 'linear_combination'.
        model_type: 'line_segment' or 'linear_combination'
        
        Returns
        --------
        fig: matplotlib object
            Figure to be drawn.
        ax: matplotlib object
           Axis object for the plots.
            
        
        """
        #cdef list argmaxs, colors
        #cdef numpy.ndarray Q_mean, X
        #cdef Py_ssize_t i, j
        #cdef object fig, ax
        
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