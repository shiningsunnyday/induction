from src.algo import nlc, mining
from src.examples import *
from src.draw.color import to_hex, CMAP
import networkx as nx
from src.config import RADIUS


def main():
    g = load_cora()
    # g = create_test_graph(1)
    # g = debug()
    grammar, model = nlc.learn_grammar(g)
    # grammar, model = mining.learn_stochastic_grammar(g)    
    model.generate(grammar)

if __name__ == "__main__":    
    breakpoint()
    main()
