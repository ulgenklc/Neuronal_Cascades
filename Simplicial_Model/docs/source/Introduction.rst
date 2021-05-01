Introduction
============
``Complex_Contagions`` is a python module to run our experiments. In this contagion model, we want to investigate the dynamics of a contagion starting from a seed cluster and spreading across the underlying network. The model and hence the package is as general as possible in a way that one can play with the parameters to obtain different network topologies and contagion models.

Geometric and Noisy Geometric Networks
******************************************************

A geometric network is a set of nodes and edges where the nodes connected to their 'close' neighbors in a euclidean distance manner.

Noisy geometric networks are obtained by adding 'noise' or edges that connects 'distant' nodes to the geometric networks. These network topology manipulations are shown to be demonstrated various contagion spread phenomenans such as wavefront propagation(WFP) or appearance of new clusters(ANC) in these networks.

Contagion Model
*********************
We are inspired by a neuoronal contagion model to asses this two phenomenans. The core function that we run our experiments decides if a given neuron is going to fire or not by a sigmoid function $$f(x) = /frac{1}{1+\exp^{-C.x}} $$. The main class we use ``geometric_network`` comes with several methods that we can manipulate the nature of the contagion very easily. For example, one can run either a stochastic or deterministic model by varying the parameter $C$ or users have the option to choose if neuorns are going to have a refractory period that they are not allowed to fire right after a spike.