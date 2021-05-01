class neuron():
    
    """
    Neuron objects corresponding to the nodes of ``Geometric_Brain_Network``.
        
    Attributes
    ----------
    neuron.name: int
        Neuron ID.
    neuron.state: int
        State of a neuron, can be 0,1 (or -1 if ``rest`` is nonzero).
    neuron.memory:int
        Memeory of a neuron. Once a neuron is activated, it is going to stay active ``memory`` many more discrete time steps(so ``memory + 1`` in total). 
    neuron.rest: int
        Refractory peiod of a neuron in terms of discrete time steps.
    neuron.threshold: int
        Threshold of a neuron, resistance to excitibility.
    neuron.history: list
        History of a neuron encoding the states that it has gone through.
    
    Parameters
    -----------
    name: str
        Neuron ID
    state: int
        State of a neuron(1 Active, 0 Inactive and -1 rest, refractory).
    memory:int
        Number of discrete time steps a neuron is going to stay active once it is activated.
    rest: int
        Refractory period of a neuron in discrete time steps.
    threshold: float
        Threshold of a neuron.
    """
    
    #cdef public int name, state, memory, rest
    #cdef public float threshold
    #cdef public list history
    
    def __init__(self, name, state, memory, rest, thresold):#int name, int state = False, int memory = 0, int rest = 0, float threshold = 0.1):
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