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
    Geometric Brain Network object to run simplicial cascades on.
        
    Attributes
    ----------
    Geometric_Brain_Network.N : int
        Size, number of nodes in the network.
    Geometric_Brain_Network.GD : int
        Geometric degree of the network.
    Geometric_Brain_Network.nGD : int
        non-Geometric degree of the network.
    Geometric_Brain_Network.manifold : str
        The geometric topology of the network. Only 'Ring' is available currently.
    Geometric_Brain_Network.text : str
        Summary of the network.
    Geometric_Brain_Network.A : array ``n x n``
        Adjacency matrix of the graph.
    Geometric_Brain_Network.nodes : List
        A list of ``neuron`` objects that corresponds to the nodes of the network in which IDs of the neurons match with the IDs of the nodes.
    Geometric_Brain_Network.time : int
        An intrinsic time property to keep track of number of iterations of the experiments.
    Geometric_Brain_Network.triangles : dict
        A dictionay of the triangles of the network where keys are node ids and values are lists of pairs of node ids that makes up a triangle together with the key value.
    Geometric_Brain_Network.higher_order : Boolean
        Flag if a higher-order experiment is to be run, that is K>0.
    
    Parameters
    -----------
    size : int
        Size of the network to be initialized.
    geometric_degree : int
        Uniform number of local neighbors that every node has.
    nongeometric_degree : int
        Fixed number of distant neighbors that every node has. 
    manifold : str
        Type of the network to be created. If 'Ring' or 'random_Ring' then a syntehtic ring network will be created, if 'lattice' then `matrix` argument can be used to input any adjacency matrix.
    noise_type : str
        k-regular or er-like
    matrix : array-like
        Argument for inputting and adjacency matrix, ff `manifold` is 'lattice'.
    perturb : int
        Number of edges per vertex that the local manifold is to be perturbed.
    higher_order : Boolean
        Flag for higher-order experiments. Since extracting the triangles is costly, when running an edge-based model, we don't have to compute them.
        
    """
    
     def __init__(self, size, geometric_degree = 1, nongeometric_degree = 0, manifold = 'Ring', noise_type = 'k-regular', matrix = None, perturb = 0, higher_order = False):
        
        self.N = size  
        self.GD = geometric_degree
        self.nGD = nongeometric_degree
        self.manifold = manifold
        self.noise = noise_type
        self.perturb = perturb
        self.higher_order = higher_order
        
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
        if self.higher_order:
            self.triangles = self.return_triangles(self.A)
        
    def get_neurons(self, neurons):
        
        """
        Sometimes we want to run experiments on a fixed network without changing the network connectivity. In this case, we can initialize a new set of neurons and use this method to inherit them in the network-- changing only the neuronal properties but not the connectivity.
        Parameters
        --------------
        neurons : list
            A list of ``neuron`` objects.
        
        Raises
        --------
        ValueError
            If the number of neurons and the size of the network doesn't match.
        
        """

        if len(neurons) != self.N: 
            raise ValueError('Size of the network and the number of neurons should match')
        
        self.nodes = neurons
        for i in range(self.N):
            self.nodes[i].neighborhood = list(np.nonzero(self.A[i])[0])
    
    def make_geometric(self):
        """
        Method for creating a geometric ring network. Options are either 'Ring' or 'random_Ring'. This will be called upon initialization automatically.
        """
        
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
    
    def add_noise_to_geometric(self):
        """
        This method adds non-geometric edges to the network that are long range. Every node will have ``nGD`` many
        nongeometric, long range, edges. Options are 'k-regular', 'ER_like' and '2D_k-regular'.
            
        """
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
            for i in range(self.nGD):
                flag1 = True
                stubs = np.arange(self.N, dtype = DTYPE)
                count = 0
                while flag1:
                    flag2 = True
                    while flag2:
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
        
    def ablate_geo_triangles(self, A):
        """
        Helper to remove links from the geometric strata. `self.perturb` many links per vertex will be removed.
        Parameters
        ============
        A : array
            Adjacency matrix of the network.
        Returns
        ===========
        A : array
            Perturbed adjacency matrix.
        """
        tris = self.return_triangles(A)
        links_to_be_removed = []
        for n in range(self.N):
            flag1 = True
            while flag1:
                flag1 = False
                random_tri = random.choices(tris['%d'%n], k = self.perturb)
                for i, e in enumerate(random_tri):
                    m = random.choice(e)
                    if [n,m] in links_to_be_removed or [m,n] in links_to_be_removed:
                        flag1 = True
                    else:
                        links_to_be_removed.append([n, m])

        for i, e in enumerate(links_to_be_removed):
            A[e[0], e[1]] = 0
            A[e[1], e[0]] = 0
        return(A)
    
    def get_nonunique_triangle_list(self, A):
        """
        Helper method finding all of the triangles in the network including the repeated ones.
        
        Parameters
        ============
        A : array
            Adjacency matrix of the network
        Returns
        ===========
        triangle_list : array
            All triangles in the network.
        """
    
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

    def get_nodes_unique_triangles(self, nonunique_triangle_list, i):
        """
        Helper function for finding triangles that flags if a triangle is repeated.
        
        Parameters
        =================
        nonunique_triangle_list : array
            Output of `get_nonunique_triangle_list`.
        i : int
            Index of the triangle whose repeated triangle neighbors to be removed.
            
        Returns
        ===========
        nonunique_triangle_list[tri_flag,1:] : array
            Removed triangles for a given node.
        tri_flag : int
            flag
        
        """
        
        tri_flag = nonunique_triangle_list[:,0]==i # make a flag for triangles using node i
        tri_flag = tri_flag * (nonunique_triangle_list[:,1]<nonunique_triangle_list[:,2])# keep only indices in ascending order

        return(nonunique_triangle_list[tri_flag,1:], tri_flag)

    def return_triangles(self):
        """
        Function for getting the triangles in the network. This will be automatically called upon initialization of ``Geometric_Brain_Network``.
        
        Returns
        --------- 
        triangles : dict
            A dictionay of the triangles of the network where keys are node ids and values are lists of pairs of node ids that makes up a triangle together with the key value.
            
        """
        
        nonunique_triangle_list = self.get_nonunique_triangle_list(self.A)
        triangles = {}
        for i in range(self.N):
            triangles[str(i)] = [list(p) for p in self.get_nodes_unique_triangles(nonunique_triangle_list,i)[0]]
        return(triangles)
    
    def neighbor_input(self, node_id, K):
        """
        This is a key function as it computes the current input from neighbors of a given node, v_{i}.
        
        Parameters
        ---------------
        node_id : int
            ID of the node whose input is going to be calculated.
        K : float
            Constant for weighing the edge and triangle activations.
        
        Returns
        ---------------
        F : float
            Neighboring neuronal input.
            
        """
        nbhood = self.nodes[node_id].neighborhood
        
        active_hood = [e for e in nbhood if self.nodes[e].state == 1]
        
        if self.higher_order:
            active_triangles = [f for f in self.triangles['%d'%node_id] if self.nodes[f[0]].state == 1 and self.nodes[f[1]].state == 1] 

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
                one_simplicies = (len(active_hood)/len(nbhood))*(1-K)
                two_simplicies = K*(len(active_triangles)/len(self.triangles['%d'%node_id]))
            
            F = one_simplicies + two_simplicies - self.nodes[node_id].threshold 
                        
        else:
            if len(nbhood) == 0:
                one_simplicies = 0
                
            else:
                one_simplicies = (len(active_hood)/len(nbhood))
                
            F = one_simplicies - self.nodes[node_id].threshold 

        return(F)
    
    def sigmoid(self, node_id, C, K):
        """
        Sigmoid function which adjusts the stochasticity of the neurons depending on `C`.
        
        Parameters
        ------------
        node_id : int
            ID of the node whose input is going to be calculated.
        C : int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        K : float
            Constant for weighing the edge and triangle activations.
        
        Returns
        ------------
        Z : float
            Probability of firing.
        """

        F = self.neighbor_input(node_id, K)
        if F == 0: 
            F = -0.1
        Z = 1/(1+np.exp(-C*F))
        
        return(Z)
    
    def update_history(self, node_id, C, K):
        """
        Helper method to update the history of the ``neuron`` objects at every iteration.
        
        Parameters
        ------------
        node_id : int
            ID of the node whose history is going to be updates.
        C : int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        K : float
            Constant for weighing the edge and triangle activations.
        """
        rand = random.uniform(0,1)
        Z  = self.sigmoid(node_id, C, K)

        if rand <= Z:
                    
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
        excited : list
            List of active neurons at the current ``time``.
        ready_to_fire : list
            List of neurons that are in the inactive state and ready to fire at ``time+1```.
        rest : list
            List of neurons that doesn't belong to either of those categories. This is empty as long as there are no refractory period.
        """
        
        for node in self.nodes:
            node.state = int(node.history[self.time])
        
        excited = [node.name for node in self.nodes if node.state == 1]
            
        ready_to_fire = [node.name for node in self.nodes if node.state == 0]
            
        rest = [node.name for node in self.nodes if node.state == -1]
        
        return(excited, ready_to_fire, rest)
                
    def initial_spread(self, seed):
        """
        Helper method to activate the neighbors of the seed node with probablity 1.
        
        Parameters
        -----------
        seed : int
            Node ID of the seed node.
        """

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
        """
        Helper method for setting the network time and tolerance to 0. This is necessary between different experiments for any set of parameters including ``seed``. Also, calls ``refresh_history`` which clears ``neuoron`` histories.
        
        Returns
        ---------
        tolerance : int
            Tolerance for experiments getting stuck at some point during contagion. Set to 0 at every trial.
        """

        self.time = 0
        tolerance = 0
        
        for node in self.nodes:
            node.refresh_history()
            
        return(tolerance)
    
    def run_dynamic(self, seed, TIME, C, K):
        """
        Core function that runs the experiments. There are couple control flags for computational efficiency. If ``self.time`` exceeds ``TIME``, flag. If there is no active neurons left in the network, flag. If everything gets activated once, flag. If ``tolerance`` exceeds 10, flag i.e. network repeats the exact state of itself 10 times.
        
        Parameters
        -----------
        seed : int
            Node ID of the seed node.
        TIME : int
            A limit on the number of iterations.
        C : int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        K : float
            Constant for weighing the edge and triangle activations.
        Returns
        -------------
        activation_times : array
            Activation times of all the nodes for contagions starting from seed.
        size_of_contagion : array
            Number of active nodes at every iteration.
        number_of_clusters :
            Number of distinct cascade clusters.
        
        """

        tolerance = self.refresh()
        activation_times = np.ones(self.N, dtype = np.int64)*TIME
        number_of_clusters = np.ones((1, TIME))
        size_of_contagion = [int(0)]
        self.initial_spread(seed)
        excited_nodes, ready_to_fire_nodes, resting_nodes = self.update_states()
        
        self.time = 1
        
        while self.time < TIME and 0 < len(excited_nodes) and np.any(activation_times==TIME) and tolerance < 10:
            size_of_contagion.append(len(excited_nodes))
            
            activation_times[excited_nodes] = np.minimum(activation_times[excited_nodes], np.array([self.time]*len(excited_nodes)))
            
            for node in ready_to_fire_nodes: 
                self.update_history(node, C, K)
                
            flag_1 = len(excited_nodes)
            excited_nodes, ready_to_fire_nodes, resting_nodes = self.update_states()
            
            if flag_1 == len(excited_nodes):
                tolerance = tolerance + 1
            
            self.time = self.time + 1
            
        for t in range(len(size_of_contagion)-2):
            number_of_clusters[0][t] = int((np.diff(self.stack_histories(len(size_of_contagion)).T[t])!=0).sum()/2)
        
        if len(size_of_contagion) < TIME:
            for i in range(len(size_of_contagion), TIME):
                size_of_contagion.append(size_of_contagion[-1])

        return(activation_times, np.array(size_of_contagion), number_of_clusters)
    
    def stack_histories(self,TIME):
        """
        Helper function for equalizing, stacking, the lengths of histories of ``neuron``s. Comes handy for visualizing single experiments.
        
        Parameters
        ------------
        TIME : int
            Number of discrete time steps for neuron histories to be visualized
        Returns
        --------
        all_history : array
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
    
    def make_distance_matrix(self, TIME, C, K):
        """
        Main function if you are running experiments for a full set of seed nodes. This creates an activation matrix by running 
        the contagion on starting from every node and encoding the first activation times of each node. Then,
        finding the euclidean distances between the columns of this matrix, creating a distance matrix so that
        the (i,j) entry corresponds to the average time(over the trials) that a contagion reaches node j starting 
        from node i.
        
        Parameters
        ------------
        TIME : int
            A limit on the number of iterations.
        C : int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.

        K : float
            Constant for weighing the edge and triangle activations.
        
        Returns
        ---------
        distance_matrix : array
            ``n x n`` array with entries the activation times of contagions starting from node i reaching to node j.
        Q : array
            ``n x t`` array with entries number of active nodes at every time step for contagions starting at different seeds.
        
        """
        D = np.zeros((self.N, self.N), dtype = np.int64)
        Q = np.zeros((self.N, TIME), dtype = np.int64)
        A = np.zeros((self.N, TIME), dtype = np.int64)
        
        for seed in range(self.N):
            D[seed], Q[seed,:], A[seed,:] = self.run_dynamic(seed, TIME, C, K)
        
        distance_matrix = euclidean_distances(D.T)
        
        return(distance_matrix, Q, A)
    
    def compute_persistence(self, distances, dimension, spy):
        
        """
        Helper to compute persistent homology using the distance matrix by building a Rips filtration up to given 
        dimension(topological features to be observed are going to be one less dimensional at max).
        First normalizes the distances before the computation.
        
        Parameters
        ----------
        distances : n x n array
            distance matrix. First output of the ``make_distance_matrix``.
        dimension: int
            Max dimension of the topological features to be computed.
        spy : bool, optional
            Take a peak at the persistence diagram.
        Returns
        -------
        Delta_min : array
            Difference of the lifetimes between longest and second longest living two 1-cycles.
        Delta_max : array
            Difference of the lifetimes between longest and shortest living two 1-cycles.
        """
        
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
    
    def display_comm_sizes(self,Q, labels, TIME, C, threshold, K):
        """
        Helper to visualize the size of the active nodes during the contagion. Shades are indicating the max 
        and min values of the spread starting from different nodes, seed node variations.
    
        Parameters
        ----------
        Q : list, [n x T+1 array]
            Output of the make_distance_matrix appended in a list
        labels: list
            Figure labels corresponding to every list element for different thresholds.
        TIME:int
            A limit on the number of iterations.
        C : int
            Constant for tuning stochasticity. Higher values yield a deterministic model whereas lower values yield a stochastic model.
        K : float
            Constant for weighing the edge and triangle activations.
        Returns
        --------
        fig : matplotlib object
            Figure to be drawn.
        ax : matplotlib object
           Axis object for the plots.
            
        
        """
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
            
        ax.set_title('%s, T = %d, C = %d,  K = %.1f, Threshold = %.2f'%(self.text, TIME, C, K, threshold), fontsize = 25)
        ax.set_xlabel('Time', fontsize = 20)
        ax.set_ylabel('Number of Active Nodes', fontsize = 20)
        ax.legend()
        return(fig,ax)
    
    
class neuron(Geometric_Brain_Network):
    
    """
    Neuron objects corresponding to the nodes of ``Geometric_Brain_Network``. This is a subclass of ``Geometric_Brain_Network``.
        
    Attributes
    ----------
    neuron.name : int
        Neuron ID.
    neuron.state : int
        State of a neuron, can be 0,1 (or -1 if ``rest`` is nonzero).
    neuron.memory : int
        Memeory of a neuron. Once a neuron is activated, it is going to stay active ``memory`` many more discrete time steps(so ``memory + 1`` in total). 
    neuron.rest : int
        Refractory peiod of a neuron in terms of discrete time steps.
    neuron.threshold : int
        Threshold of a neuron, resistance to excitibility.
    neuron.history : list
        History of a neuron encoding the states that it has gone through.
    
    Parameters
    -----------
    name : str
        Neuron ID
    state : int
        State of a neuron(1 Active, 0 Inactive and -1 rest, refractory).
    memory : int
        Number of discrete time steps a neuron is going to stay active once it is activated.
    rest : int
        Refractory period of a neuron in discrete time steps.
    threshold : float
        Threshold of a neuron.
    """
    
    def __init__(self, name, state, memory, rest, thresold):
        self.name = name
        self.state = state
        self.memory = memory
        self.rest = rest
        self.threshold = threshold
        
        self.refresh_history()
        
    def refresh_history(self):
        """
        Helper method that sets the history of every neuron to an empty list. It is called after ``refresh``.
        """
        self.history = []
