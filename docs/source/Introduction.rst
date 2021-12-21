Introduction
==============
Communication property of the brain often comes into surface as in the form of synchronization, self-organization and stochastic collective dynamics. Indeed, in the most homogeneosly distributed inch of the brain, networks of cortical neurons receive, process and transfer large number of synaptic responses with great temporal precision to maintain cognitive activity. 

Integration and propagation of such information occur via cascading activity of groups of neurons arising from the interactions between recent activity history, non-linear neuronal properties and network topology.

Here, we develop theoretical framework for the interplay between the dynamics and network topology manifesting through higher-order connections embedded in a manifold structure for neuronal activity.

``Neuronal Cascades`` is the python module that we run our experiments. In this module, one can investigate the dynamics of a simplicial threshold model (STM) starting from a seed cluster and spreading across the underlying simplicial complex. The model and hence the package is as general as possible in a way that one can play with the parameters to obtain different network topologies and cascade models.


Geometric and Noisy Geometric complexes
******************************************************

A substantial fraction of the synaptic input to a cortical neuron comes from nearby neurons within local circuits, while the remaining synapses carry signals from more distant locations. Therefore, geometric networks are traditionally used as a proxy to mimic network topology of cortical brain regions.

A geometric network is a set of nodes and edges where the nodes connected to their 'close' neighbors in a euclidean distance manner.

Noisy geometric networks are obtained by adding 'noise' or edges that connects 'distant' vertices of the geometric network. These network topology manipulations are shown to demonstrate various contagion spread phenomenans such as wavefront propagation(WFP) or appearance of new clusters(ANC) in these networks. 

We study the spatio-temporal patterns of STM cascades over noisy geometric complexes, which contain both short- and long-range simplices and are a generalization of noisy geometric networks.

.. figure:: ring_WFP.pdf
   :width: 200px
   :height: 200px
   :scale: 300 %
   :align: center
   
   A noisy ring complex involves vertices that lie along a 1D manifold that is embedded in a 2D ambient space. (Vertices are placed slightly alongside the manifold to allow easy visualization of 2-simplices.)  Each vertex has :math:`d^{(G)} = 4` geometric edges to nearby vertices and :math:`d^{(NG)} = 1` nongeometric edge to a distant vertex. Higher-dimensional simplices arise in the associated clique complex and are similarly classified.  An STM cascade exhibits WFP when it progresses along the ring manifold, and ANC events when it jumps across a long-range edge or higher-dimensional simplex.


Neuronal Subtypes
****************************

Intracellular recordings show that cortical neurons display beyond pairwise dynamics and are subjected to an intense synaptic bombardment working in a high-conductance state.

In the package, ``neuron`` objects can have individual activation thresholds as well as memory and refractory periods as a function of discrete time steps. This generalization enables heterogenity in the experiments as well as complexity of the non-trivial interactions.


Simplicial Threshold Model
************************************
Processing the frequent chatter of neurons necessitates a neuronal activation rule that is prone to intercellular noise to keep the neuronal communication on a thin line between dynamic states.

We are inspired by a neuoronal cascading model to asses this phenomena. The core function that we run our experiments decides if a given neuron is going to fire or not by a sigmoid function :math:`f(R_{i},C) = \frac{1}{1+\exp^{-C.R_{i}}}` where :math:`R_{i}`, the simplicial exposure, is a function of current network history defined by :math:`R_{i} = \left[(1-K)*\sum_{e \in E_{i}} \frac{e}{d_{i}^{e}} + (K)*\sum_{t \in T_{i}}\frac{t}{d_{i}^{t}}\right] - \tau_{i}` where :math:`E_{i}` is the set of active edge neighbors, :math:`T_{i}` is the set of active triangle neighbors of node :math:`i`, :math:`d_{i}^{e}` and :math:`d_{i}^{t}` are edge and triangle degrees of node :math:`i` respectively. The constant :math:`K` ,2-simplex influence, is used to strike a balance between traditional activation maps and higher order, or simplicial, cascade maps.

.. figure:: response.jpg
   :width: 200px
   :height: 200px
   :scale: 300 %
   :align: center
   
   Set of neuronal activation functions as a function of :math:`C`.


The main class we use ``Geometric_Brain_Network`` comes with several methods that we can manipulate the nature of the contagion very easily. For example, one can run either a stochastic or deterministic model by varying the parameter :math:`C`. Moreover, :math:`K=0` recovers an edge contagion whereas :math:`K=1` recovers a pure triangle contagion.

.. figure:: active_triangles.pdf
   :width: 200px
   :height: 200px
   :scale: 300 %
   :align: center
   
   Each k-simplex has a binary state :math:`x^{k}_{i}(t)\in \{0,1\}` indicating whether it is inactive or active, respectively, at time t.  Active k-simplices influence inactive boundary vertices, possibly causing them to become active at the next time step. The dimension of an STM cascade refers to the highest-dimension k-simplex that is active, and we focus herein on 2D STM cascades.