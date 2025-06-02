from src.config import METHOD, DATASET, GRAMMAR
import importlib
from src.examples import *
from src.draw.color import to_hex, CMAP
from src.draw.graph import draw_graph
from src.config import RADIUS
from argparse import ArgumentParser
import pickle
from src.grammar.common import get_args
from src.grammar.utils import *
import os
import sys
import time
import threading
# from src.model import graph_regression, transformer_regression

def start_ckpt_watchdog(ckpt_path: str, timeout_sec: int = 600, poll_interval: int = 60):
    """
    Monitors the modification time of `ckpt_path`. If the file isn't created
    or hasn't been updated for `timeout_sec` seconds _since job start or
    since last update_, the process will exit(1).
    """
    start_time = time.time()
    def _watchdog():
        # initialize to the later of job-start or first ckpt write
        last_mtime = start_time
        while True:
            try:
                if os.path.exists(ckpt_path):
                    mtime = os.path.getmtime(ckpt_path)
                    # if file mtime is newer than our marker, bump it
                    if mtime > last_mtime:
                        last_mtime = mtime
                # if we’ve gone > timeout since last_mtime, bail out
                if time.time() - last_mtime > timeout_sec:
                    sys.stderr.write(
                        f"[ckpt-watchdog] no update to {ckpt_path} for {timeout_sec}s, exiting.\n"
                    )
                    os._exit(1)
            except Exception as e:
                sys.stderr.write(f"[ckpt-watchdog] error: {e}\n")
            time.sleep(poll_interval)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


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
        g = load_ckt(args)
    elif DATASET == "enas":
        g = load_enas(args)
    elif DATASET == "bn":
        g = load_bn(args)
    elif DATASET == "mol":
        g = load_mols(args)
    else:
        raise NotImplementedError
    return g


def main(args):
    g = load_data(args)
    if 'learn' in args.task:
        # if os.path.exists(os.path.join(IMG_DIR, f"grammar-{args.dataset}-{args.seed}.pkl")):
        #     return
        gr, model = grammar.learn_grammar(g, args)
        # grammar, model = nlc.learn_grammar(g)
        # grammar, model = mining.learn_stochastic_grammar(g)

    if 'prediction' in args.task:       
        samples = gr.induce(model)
        for i in range(len(samples)):
            draw_graph(samples[i], os.path.join(IMG_DIR, f"{i}_model.png"))
        # graph_regression(samples)
        # transformer_regression(samples)
    
    if 'generate' in args.task:
        path = os.path.join(IMG_DIR, f"smiles-{args.mol_dataset}-{args.seed}.txt")        
        # gr, _, _ = pickle.load(open(f'{os.getcwd()}/cache/api_ckt_ednce/12.pkl', 'rb'))
        gr = pickle.load(
            open(os.path.join(IMG_DIR, f"grammar-{args.mol_dataset}-{args.seed}.pkl"), "rb")
        )
        samples = gr.generate(args)
        # path = f"{os.getcwd()}/data/api_ckt_ednce/samples.txt"
        write_file(samples, path)
        # convert_and_write(samples, path)


if __name__ == "__main__":
    args = get_args()
    if args.grammar_ckpt:
        # start monitoring your ckpt file:
        start_ckpt_watchdog(args.grammar_ckpt, timeout_sec=5*60, poll_interval=60)    
    main(args)
