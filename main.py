from src.algo import nlc
from src.examples import *

if __name__ == "__main__":
    g = create_test_graph(1)
    grammar = nlc.learn_grammar(g)