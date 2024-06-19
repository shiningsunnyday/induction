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


def load_data():
    if DATASET == 'cora':
        g = load_cora()
    elif DATASET == 'test':
        g = create_test_graph(1)
    elif DATASET == 'debug':
        g = debug()    
    else:
        raise NotImplementedError
    return g



def main(args):    
    g = load_data()    
    gr, model = grammar.learn_grammar(g)
    # grammar, model = nlc.learn_grammar(g)
    # grammar, model = mining.learn_stochastic_grammar(g)    
    model.generate(gr)
    samples = gr.generate()
    gen_dir = os.path.join(IMG_DIR, "generate/")
    os.makedirs(gen_dir, exist_ok=True)
    for i, g in enumerate(samples):
        draw_graph(g, os.path.join(gen_dir, f'graph_{i}.png'))                
    breakpoint()


if __name__ == "__main__":        
    parser = ArgumentParser()
    args = parser.parse_args()
    breakpoint()
    main(args)
