from src.config import METHOD, DATASET, GRAMMAR, IMG_DIR
import importlib
if 'api' in METHOD:
    grammar = importlib.import_module(f"src.algo.{GRAMMAR}")    
else:
    grammar = importlib.import_module(f"src.algo.mining.{GRAMMAR}")
from src.examples import *
from src.draw.color import to_hex, CMAP
from src.draw.graph import draw_graph
import os
from src.config import RADIUS
from argparse import ArgumentParser
import pickle


def load_data():
    if DATASET == 'cora':
        g = load_cora()
    elif DATASET == 'test':
        g = create_test_graph(1)
    elif DATASET == 'debug':
        g = debug()   
    elif DATASET == 'house':
        g = create_house_graph()
    elif DATASET == 'ckt':
        g = load_ckt()
    else:
        raise NotImplementedError
    return g



def main(args):    
    g = load_data() 
    # gr, _, _ = pickle.load(open(f'{os.getcwd()}/cache/api_ckt_ednce/9.pkl', 'rb'))
    gr, model = grammar.learn_grammar(g)
    # grammar, model = nlc.learn_grammar(g)
    # grammar, model = mining.learn_stochastic_grammar(g)    
    # model.generate(gr)
    breakpoint()    
    samples = gr.generate()
    


if __name__ == "__main__":        
    parser = ArgumentParser()
    args = parser.parse_args()
    breakpoint()
    main(args)
