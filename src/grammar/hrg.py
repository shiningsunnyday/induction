class HG():
    """
        There is a global vocabulary of labels C, each with a type in natural numbers N.
        A hypergraph H over a vocabulary of labels C is defined as a tuple (V, E, att, lab, ext) where:
        - V := finite set of nodes
        - E: finite set of hyperedges
        - att: E -> V^* := mapping assigning a sequence of pairwise distinct attachment nodes to each e in E
        - lab: E -> C := mapping that labels each hyperedge s.t. type(lab(e)) = |att(e)|
        - ext in V^* := sequence of pairwise distinct external nodes
    """
    def __init__(self):
        self.V = []
        self.E = []
        self.ext = []
    

    def add_node(self):
        breakpoint()



if __name__ == "__main__":
    hg = HRG()
