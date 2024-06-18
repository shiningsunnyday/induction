from src.algo import nlc
from src.examples import *
from src.draw.color import to_hex, CMAP
import networkx as nx
from src.config import RADIUS
from argparse import ArgumentParser


def load_data(data):
    if data == 'cora':
        g = load_cora()
    elif data == 'test':
        g = create_test_graph(1)
    elif data == 'debug':
        g = debug()    
    else:
        raise NotImplementedError
    return g



def main(args):    
    g = load_data(args.data)
    grammar, model = nlc.learn_grammar(g)
    # grammar, model = nlc.learn_grammar(g)
    # grammar, model = mining.learn_stochastic_grammar(g)    
    model.generate(grammar)

if __name__ == "__main__":        
    parser = ArgumentParser()
    parser.add_argument('--data', choices=['cora', 'test', 'debug'])
    args = parser.parse_args()
    main(args)
