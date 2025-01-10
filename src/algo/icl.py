from src.config import *
from src.algo.hg import llm_call
import rdkit.Chem as Chem

def learn_grammar(smiles, args):
    path = f"data/{args.mol_dataset}/api_mol_icl.txt"    
    text = '\n'.join(open(path).readlines())
    text = text.replace('<optional>', '\n'.join(smiles))    
    path = f'data/api_mol_icl-{args.mol_dataset}.txt'
    if os.path.exists(path):
        res = open(path).readlines()
        res = [l.rstrip('\n').strip('\n') for l in res]
        print("loaded len(res)", len(res))
    else:
        res = []
    while len(res) < args.num_samples:
        print("len(res)", len(res))
        smiles = llm_call([], None, prompt=text)        
        for smi in smiles.split('\n'):
            if len(smi) == 0:
                continue
            res.append(smi)
            if len(res) == args.num_samples:
                break
        with open(path, 'w+') as f:
            for s in res:
                f.write(s+'\n')
