from src.config import METHOD, DATASET, GRAMMAR
import importlib
from src.examples import *
from src.draw.color import to_hex, CMAP
from src.draw.graph import draw_graph
from src.config import RADIUS
from argparse import ArgumentParser
import pickle
from src.grammar.common import get_args


def load_data(args):
    if DATASET == "cora":
        g = load_cora()
    elif DATASET == "test":
        g = create_test_graph(1)
    elif DATASET == "debug":
        g = debug()
    elif DATASET == "house":
        g = create_house_graph()
    elif DATASET == "ckt":
        g = load_ckt()
    elif DATASET == "mol":
        g = read_file(
            f"/home/msun415/induction/data/api_mol_hg/{args.dataset}_smiles.txt"
        )
    else:
        raise NotImplementedError
    return g


def main(args):
    g = load_data(args)    
    if 'learn' in args.task:
        if os.path.exists(os.path.join(IMG_DIR, f"grammar-{args.dataset}-{args.seed}.pkl")):
            return
        gr, model = grammar.learn_grammar(g, args)
        # grammar, model = nlc.learn_grammar(g)
        # grammar, model = mining.learn_stochastic_grammar(g)        
    # model.generate(gr)
    
    if 'generate' in args.task:
        path = os.path.join(IMG_DIR, f"smiles-{args.dataset}-{args.seed}.txt")        
        # gr, _, _ = pickle.load(open(f'{os.getcwd()}/cache/api_ckt_ednce/12.pkl', 'rb'))
        gr = pickle.load(
            open(os.path.join(IMG_DIR, f"grammar-{args.dataset}-{args.seed}.pkl"), "rb")
        )
        samples = gr.generate(args)
        # path = f"{os.getcwd()}/data/api_ckt_ednce/samples.txt"
        write_file(samples, path)
        # convert_and_write(samples, path)


if __name__ == "__main__":
    args = get_args()
    main(args)
