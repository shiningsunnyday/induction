from src.algo import nlc
from src.examples import *
import json
import networkx as nx

def main():
    g = nx.node_link_graph(json.load(open('/home/msun415/induction/data/nx/cora.json')))
    labels = list(set([g.nodes[n]['label'] for n in g]))
    assert len(labels) == len(LABELS)
    lookup = dict(zip(labels, LABELS))
    for n in g:
        g.nodes[n]['label'] = lookup[g.nodes[n]['label']]
    grammar = nlc.learn_grammar(g)    

if __name__ == "__main__":    
    breakpoint()
    main()