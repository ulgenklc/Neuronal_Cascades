cdef class neuron:
    
    cdef public int name, state, memory, rest
    cdef public float threshold
    cdef public list history
    
    def __init__(self, int name, int state = False, int memory = 0, int rest = 0, float threshold = 0.1):
        self.name = name
        self.state = state
        self.memory = memory
        self.rest = rest
        self.threshold = threshold
        
        self.refresh_history()
        
    def refresh_history(self):
        self.history = []