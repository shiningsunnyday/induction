from src.grammar.hg import *
from src.config import *
from src.draw.mol import *
import hashlib
import time
from filelock import FileLock
import threading

def get_thread_logger():
    """
    Returns a logger whose name encodes the current thread.
    """
    thread_name = threading.current_thread().name
    logger_name = f"global_logger.{thread_name}"
    return logging.getLogger(logger_name)

def hash_dict(d):
    # Convert the dictionary into a tuple of sorted key-value pairs
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


# Define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError, openai.error.APIError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                print(e)
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay *= exponential_base * (1 + jitter * random.random())

                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper



def bradley_terry(pairwise_results, num_debaters, max_iterations=100, tolerance=1e-6):
    # Step 1: Initialize skill levels for each debater (theta values)
    theta = [1.0 for _ in range(num_debaters)]  # Initial guess, could be any positive number

    for iteration in range(max_iterations):
        theta_old = theta[:]  # Keep a copy of the old theta values for convergence check

        # Step 2: For each debater i, update their theta based on all pairwise matches
        for i in range(num_debaters):
            numerator = 0.0
            denominator = 0.0

            # Step 3: Iterate over all match results involving debater i
            for (i_prime, j, S_i_prime) in pairwise_results:
                if i_prime == i:  # i played against j
                    numerator += S_i_prime  # Sum of debater i's scores
                    denominator += S_i_prime + (1 - S_i_prime) * (theta[j] / theta[i])  # Bradley-Terry denominator
                elif j == i:  # i played against i_prime
                    numerator += (1 - S_i_prime)  # Sum of debater j's scores
                    denominator += (1 - S_i_prime) + S_i_prime * (theta[i_prime] / theta[i])

            # Step 4: Update theta for debater i using the Bradley-Terry scaling formula
            theta[i] = numerator / denominator

        # Step 5: Check for convergence
        if max(abs(theta[i] - theta_old[i]) for i in range(num_debaters)) < tolerance:
            break

    # Step 6: Rank debaters by theta values (higher is better)
    ranked_debaters = sorted(range(num_debaters), key=lambda x: theta[x], reverse=True)

    return ranked_debaters[0]  # The winner is the debater with the highest theta value




def get_rule_multisets(grammars):
    smile_set = set(grammars[0])
    for i in range(len(grammars)):
        assert set(grammars[i].mol_lookup) == smile_set
    smile_set = set(smis_1)
    rule_multisets = defaultdict(list)
    for smi in list(smile_set):
        for grammar, trees, smis in [(globals()[f'grammar_{seed}'], globals()[f'trees_{seed}'], globals()[f'smis_{seed}']) for seed in range(1,6)]:    
            idx = smis.index(smi)    
            rule_gs = []
            for i in range(len(trees[idx])):
                data = trees[idx].nodes[i]
                symbol = data['symbol']
                rule_str = data['rule']
                rule_idx = grammar.rule_idx_lookup[smi][symbol][rule_str]
                rule = grammar.hrg.rules[rule_idx]
                rhs_g = rule.rhs.visualize(path='', return_g=True)
                rule_gs.append(rhs_g)
            rule_multisets[smi].append(rule_gs)





@retry_with_exponential_backoff
def create_chat_completion(return_logprobs=False, **kwargs):
    hash_ = hash_dict(kwargs)
    cache_path = os.path.join(IMG_DIR, f"{hash_}.txt")
    # if os.path.exists(cache_path):
    if CACHE:
        res = open(cache_path).read()
    else:
        completion = openai.ChatCompletion.create(**kwargs)
        res = completion.choices[0].message.content
        if return_logprobs:
            logprobs = completion['choices'][0]['logprobs']
            return logprobs['content'][0]['top_logprobs'], res
        if CACHE:
            with open(cache_path, "w+") as f:
                f.write(res)
    return res


def llm_call(img_paths, prompt_path, optional_prompts=[], prompt=None, return_logprobs=False):
    """
    This function uses prompt read from prompt_path and a list of img content.
    Parameters:
        img_paths: list of paths to img files
        prompt_path: a .txt file path
        optional_prompt: lambda function with single arg
        optional text prompt to process the output of the response
        return_logprobs: whether to return log probs of top k tokens
    Output:
        Response of call
    """
    # logger = logging.getLogger("global_logger")
    logger = get_thread_logger()
    descr = ""
    if len(img_paths):
        descr += "IMG PATHS:"
    for path in img_paths:
        descr += f" {path}"
    if prompt_path:
        descr += f"\nPROMPT PATH: {prompt_path}"
    if prompt:
        descr += f"\nPROMPT: {prompt}"
    if VERBOSE:
        logger.info(f"=====BEGIN LLM call=====\n{descr}\n")
    settings = {
        "temperature": 0,
        "seed": 42,
        "top_p": 0,
        "n": 1
    } # try to make deterministic
    settings = {}
    if return_logprobs:
        settings['top_logprobs'] = 5
        settings['logprobs'] = True
    base64_images = prepare_images(img_paths)
    if prompt is None:
        prompt = "".join(open(prompt_path).readlines())
    res = create_chat_completion(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                    for base64_image in base64_images
                ],
            }
        ],
        return_logprobs=return_logprobs,
        **settings,
    )
    if return_logprobs: 
        logprobs, res = res
        if VERBOSE:
            logger.info("===RESPONSE===")
            logger.info(res + "\n")
            logger.info("=====END LLM call=====")
        return logprobs
    if VERBOSE:
        logger.info("===PROMPT===")
        logger.info(prompt)
        logger.info("===RESPONSE===")
        logger.info(res + "\n")
    if len(optional_prompts):
        ans_heads = []
        for optional_prompt in optional_prompts:
            ans = create_chat_completion(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": optional_prompt(res)}],
                    }
                ],
                **settings,
            )
            if VERBOSE:
                logger.info("===PROMPT===")
                logger.info(optional_prompt(res))
                logger.info("===RESPONSE===")
                logger.info(ans + "\n")
            ans_heads.append(ans)
        if VERBOSE: logger.info("=====END LLM call=====")
        return ans_heads, res
    else:
        if VERBOSE: logger.info("=====END LLM call=====")
        return res


def llm_choose_edit(img_paths, prompt_path, prompt=None):
    post_prompt = (
        lambda res: f"I want you to perform a simple data post-processing step of the following response:\n{res}\n The input is a response from another language agent. It may or may not contain an answer in the form of a single pair. If it does, output the pair in x,y format and NOTHING ELSE. Don't include explanations or superlatives. Just output the answer. If it doesn't contain an answer in the form of a pair, output the single word NONE."
    )
    ans_heads, cot = llm_call(img_paths, prompt_path, [post_prompt], prompt=prompt)
    return ans_heads[0], cot


def llm_edit_cliques(cg, mol, prompt_path, folder, scheme='zero', ablate=False, text_only=False):
    # if folder is None:
    #     d = get_next_version(IMG_DIR)
    #     dir_name = os.path.join(IMG_DIR, f"{d}")        
    # else:
    dir_name = os.path.join(folder, 'cliques/')
    os.makedirs(dir_name, exist_ok=True)
    cots = []
    while True:
        if text_only:
            path = None
            path_indv = None
        else:
            i = get_next_version(dir_name, dir=False)
            path = os.path.join(dir_name, f"{i}.png")
            path_indv = os.path.join(dir_name, f"{i}_indv.png")
        cliques = clique_drawing(cg, mol, path, scheme=scheme, text_only=text_only)        
        clique_drawing(cg, mol, path_indv, scheme=scheme, isolate=True, text_only=text_only)
        if text_only:
            cliques, descr = cliques
            isolate_cot = llm_describe_cliques(descr, [path_indv], prompt_6_path, text_only=text_only)
        else:
            isolate_cot = llm_describe_cliques(cliques, [path_indv], prompt_6_path, text_only=text_only)
        combined_prompt = ''.join(open(prompt_path).readlines())
        combined_prompt = combined_prompt.replace("<optional>", isolate_cot)
        if ablate:
            cots.append((isolate_cot, None))
            break
        pair, cot = llm_choose_edit([] if text_only else [path, path_indv], None, prompt=combined_prompt)
        cots.append((isolate_cot, cot))
        match = re.match(f"(\d+),(\d+)", pair)
        if match:
            e1 = int(match.groups()[0])-(scheme=='one')
            e2 = int(match.groups()[1])-(scheme=='one')
            if max(e1, e2) >= len(cliques):
                break
        else:
            break
        cq = cliques[e1] + cliques[e2]
        cg.add_edges_from(product(cq, cq))
        cg.remove_edges_from(nx.selfloop_edges(cg))
    
    return cg, path, cots

def sanity_check_num_cliques(res):
    num = 0
    while f"Motif {num}" in res:
        num += 1
    return num


def llm_describe_cliques(cliques, paths, prompt_path, text_only=False):
    # logger = logging.getLogger("global_logger")
    logger = get_thread_logger()
    # numbering_str = f'0 to {len(cliques)-1})'
    # post_prompt = (
    #     lambda res: f"I want you to do a simple check of the following response:\n{res}\n The input is a response from another language agent. I want you to sanity check if it contains descriptions for PRECISELY {len(cliques)} motifs, numbered from {numbering_str}. If there is an issue with the numbering, or the response is missing some motifs, the response fails the sanity check. Output YES if the response passes the sanity check, and NO otherwise."
    # )
    prompt = ''.join(open(prompt_path).readlines())
    if text_only:
        cliques_str = '\n'.join(cliques)
        prompt = prompt.replace("<optional>", f"Here are {len(cliques)} substructures of a molecule.\n\n{cliques_str}.\n\nThey are numbered one-by-one from Motif 0 to Motif {len(cliques)-1}, inclusive. I want you to explain, concisely, what each numbered motif is. Make sure to start from Motif 0 and go in order of the numbering. MAKE SURE you describe EVERY MOTIF!")
    else:
        if len(cliques) > 1:
            prompt = prompt.replace("<optional>", f"I will highlight for you {len(cliques)} of the substructures of a molecule. They are numbered one-by-one from Motif 0 to Motif {len(cliques)-1}, inclusive. I want you to explain, concisely, what each numbered motif is. Make sure to start from Motif 0 and go in order of the numbering. MAKE SURE you describe EVERY MOTIF!")
        else:
            prompt = prompt.replace("<optional>", "I will highlight for you ONE substructure of a molecule: Motif 0. I want you to explain, concisely, what this numbered motif is. For formatting reasons, make sure to answer in the format:\nMotif 0. [insert your description]")
    tries = 0
    while True:
        if tries == MAX_TRIES:
            logger.error(f"{paths} exceeded max tries")
            sys.exit(1)
        if text_only:
            res = llm_call([], None, optional_prompts=[], prompt=prompt)
        else:
            res = llm_call(paths, None, optional_prompts=[], prompt=prompt)
        if sanity_check_num_cliques(res) == len(cliques):
            break                
        tries += 1
    return res


def llm_choose_root(img_path, prompt_path, folder, scheme='zero', text_only=False):
    post_prompt = (
        lambda res: f"I want you to perform a simple data post-processing step of the following response:\n{res}\n The input is a response from another language agent. It may or may not contain an answer in the form of a single integer. If it does, output the integer and NOTHING ELSE. Don't include explanations or superlatives. Just output the answer. If it doesn't contain an answer in the form of a pair, output the single word NONE."
    )
    motif_cot = ''.join(open(os.path.join(folder, 'motifs_cot.txt')).readlines())    
    combined_prompt = ''.join(open(prompt_path).readlines())    
    combined_prompt = combined_prompt.replace("<optional>", motif_cot)    
    if text_only:
        ans_heads, cot = llm_call([], None, optional_prompts=[post_prompt], prompt=combined_prompt)
    else:
        ans_heads, cot = llm_call([img_path], None, optional_prompts=[post_prompt], prompt=combined_prompt)      
    root = ans_heads[0]
    match = re.match("^\d+$", root)
    if match:
        return int(root)-(scheme=='one'), cot
    else:
        return 0, cot


def init_tree(cg):
    tree = nx.Graph()
    for cq in nx.find_cliques(cg):
        tree.add_node(len(tree), nodes=cq)
    for n1 in tree:
        for n2 in tree:
            nodes1 = set(tree.nodes[n1]["nodes"])
            nodes2 = set(tree.nodes[n2]["nodes"])
            if nodes1 & nodes2:
                tree.add_edge(n1, n2, weight=len(nodes1 & nodes2))
    tree.remove_edges_from(nx.selfloop_edges(tree))
    return tree


def llm_break_edge(img_path, prompt_path, prompt=None, text_only=False):
    post_prompt = (
        lambda res: f"I want you to perform a simple data post-processing step of the following response:\n{res}\n The input is a response from another language agent. It may contain an answer in the form of a single integer for the LEAST important interaction. If it does, output the integer and NOTHING ELSE. Don't include explanations or superlatives. Just output the answer. If it doesn't contain an answer in the form of a pair, output the single word NONE."
    )
    if text_only:
        ans_heads, _ = llm_call([], prompt_path, [post_prompt], prompt=prompt)
    else:
        ans_heads, _ = llm_call([img_path], prompt_path, [post_prompt], prompt=prompt)
    return ans_heads[0]


def llm_break_cycles(tree, mol, root, prompt_path, folder, scheme='zero', text_only=False):
    # logger = logging.getLogger("global_logger")
    logger = get_thread_logger()
    describe_post_prompt = (
        lambda res: f"I want you to perform a simple post-processing step of the following response:\n{res}\n The input is a response from another language agent. It describes motifs numbered from Motif 0 to Motif {len(tree)-1}, inclusive! I want you to rephrase each motif description by filling in X within the following sentence template: \nThis motif is X\n Be sure to condense the description and output a single PHRASE such that the sentence template is grammatically correct. Don't capitalize the first letter, since your answer should just be a phrase. Output your rephrasing for each motif on a SEPARATE line, using only a new line for delimiting different motifs. Don't output anything else. MAKE SURE you do it for EVERY MOTIF!"
    )
    # if folder is None:
    #     d = get_next_version(IMG_DIR)
    #     dir_name = os.path.join(IMG_DIR, f"{d}")
    # else:
    dir_name = os.path.join(folder, 'cycles/')
    os.makedirs(dir_name, exist_ok=True)
    motif_cot = ''.join(open(os.path.join(folder, 'motifs_cot.txt')).readlines())
    prompt = describe_post_prompt(motif_cot)
    tries = 0
    while True:
        if tries == MAX_TRIES:
            logger.error(f"break cycles {folder} exceeded max tries")
            sys.exit(1)
        describe_cot = llm_call([], None, prompt=prompt)
        describes = [line for line in describe_cot.split('\n') if line]
        if len(describes) == len(tree):
            break
        tries += 1
    while not nx.is_tree(tree):
        if text_only:
            path = None
        else:
            i = get_next_version(dir_name, dir=False)
            path = os.path.join(dir_name, f"{i}.png")
        cyc = nx.find_cycle(tree, root)
        if cyc:
            if text_only:
                path = None
            else:
                path = draw_cycle(cyc, tree, mol, path)
            combined_prompt = ''.join(open(prompt_path).readlines())
            interaction_descrs = [f"Interaction {i} features {describes[c[0]]} and {describes[c[1]]}." for i, c in enumerate(cyc)]
            interaction_descr = '\n'.join(interaction_descrs)
            combined_prompt = combined_prompt.replace('<optional>', interaction_descr)            
            e = llm_break_edge(path, prompt_path, prompt=combined_prompt, text_only=text_only)        
            match = re.match("^\d+$", e)
            if match:
                e = int(e)-(scheme=='one')
                if e >= len(cyc):
                    continue
                e1, e2 = cyc[e]
                tree.remove_edge(e1, e2)
        else:
            break
    return tree


def llm_describe_tree(tree, mol, root, prompt_path, folder=None):
    # post_prompt = (
    #     lambda res: f"I want you to summarize the following response into a single prose paragraph, without bullets or sections. \n{res}\n"
    # )    
    if folder is None:
        d = get_next_version(IMG_DIR)
        dir_name = os.path.join(IMG_DIR, f"{d}")
    else:
        dir_name = os.path.join(folder, 'tree/')
    os.makedirs(dir_name, exist_ok=True)
    bfs_tree = nx.bfs_tree(tree, root)
    cots = []
    for i, n in enumerate(nx.topological_sort(bfs_tree)):
        if n == root:
            # cq = tree.nodes[root]['nodes']
            # fig, ax = plt.subplots()
            # ax.axis('off')    
            # draw_cliques(None, mol, ax, [(None, cq, (1, 0, 0))], label=False)
            # fig.set_facecolor('white')
            # path = os.path.join(dir_name, f'{root}.png')
            # fig.savefig(path)
            continue        
        p = list(bfs_tree.predecessors(n))[0]
        cq_1 = tree.nodes[p]['nodes']
        cq_2 = tree.nodes[n]['nodes']
        fig, ax = plt.subplots()
        ax.axis('off')    
        draw_cliques(None, mol, ax, [(None, cq_1, (1, 0, 0)), (None, cq_2, (0, 1, 0))], label=False)
        fig.set_facecolor('white')
        path = os.path.join(dir_name, f'{p}-{n}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        cot = llm_call([path], prompt_path)
        with open(os.path.join(dir_name, f'{p}-{n}.txt'), 'w+') as f:
            f.write(cot)
        cots.append(f"{i+1}. {cot}")
    with open(os.path.join(folder, 'root_cot.txt')) as f:
        root_cot = ''.join(f.readlines())
        cots = [f"{1}. {root_cot}"] + cots
    cots_prompt = '\n'.join(cots)
    # with open(os.path.join(folder, 'all_cot.txt'), 'w+') as f:
    #     f.write(cots_prompt)
    # return llm_call([], None, optional_prompts=[], prompt=cots_prompt)
    return cots_prompt
    


def convert_to_node_set_tree(tree):
    assert len(tree) == len(
        set(tuple(sorted(tuple(tree.nodes[n]["nodes"]))) for n in tree)
    )  # make sure no repeated nodes
    # T = convert_to_directed(tree, root)
    T = defaultdict(set)
    for t_1, t_2 in tree.edges:
        nodes_1 = frozenset(tree.nodes[t_1]["nodes"])
        nodes_2 = frozenset(tree.nodes[t_2]["nodes"])
        T[frozenset(nodes_1)].add(nodes_2)
        T[frozenset(nodes_2)].add(nodes_1)
    return T


def get_next_version(fig_dir, dir=True):
    lock = FileLock(os.path.join(fig_dir, "dir.lock"))
    with lock:
        if dir:
            check = lambda f: os.path.isdir(f) and "lock" not in f
        else:
            check = lambda f: os.path.isfile(f) and "lock" not in f
        dirs = [d for d in os.listdir(fig_dir) if check(os.path.join(fig_dir, d))]
        if dir:
            versions = [int(d) for d in dirs if d.isdigit()]
        else:
            versions = [int(f.split(".")[0]) for f in dirs if f.split(".")[0].isdigit()]
        ans = max(versions) + 1 if len(versions) else 0
    return ans


def _learn_grammar(smiles, args):
    # logger = logging.getLogger("global_logger")    
    logger = get_thread_logger()
    folder = f"data/api_mol_hg/learn-{time.time()}/"
    logger.info(f"{folder} -- begin learning grammar rules for {smiles}")
    os.makedirs(folder, exist_ok=True)
    draw_smiles(smiles, path=os.path.join(folder, "smiles.png"), label_bonds=False)
    with open(os.path.join(folder, "smiles.txt"), "w+") as f:
        f.write(f"{smiles}")        
    draw_smiles(smiles, path=os.path.join(folder, "smiles_labeled.png"), label_atoms=True, label_atom_idx=True, label_bonds=True)    
    mol, cg, base_cg = chordal_mol_graph(smiles)
    if VISUALIZE:
        base_motif_path = os.path.join(folder, "base_motifs.png")        
        clique_drawing(base_cg, mol, path=base_motif_path, scheme=args.scheme)
    cg, path, cots = llm_edit_cliques(cg, mol, prompt_1_path, folder=folder, scheme=args.scheme, ablate=args.ablate_merge, text_only=args.text_only)
    with open(os.path.join(folder, 'motifs_cot.txt'), 'w+') as f:
        f.write(cots[-1][0])
    if not args.ablate_merge:
        for i, (_, clique_cot) in enumerate(cots):
            with open(os.path.join(folder, f'clique_{i}_cot.txt'), 'w+') as f:
                f.write(clique_cot)
    tree = init_tree(cg)
    if VERBOSE:
        logger.info(f"{folder} -- Initialized tree is tree? {nx.is_tree(tree)}")
    motif_path = os.path.join(folder, "motifs.png")
    clique_drawing(cg, mol, path=motif_path, isolate=True, scheme=args.scheme)
    tries = 0
    if args.ablate_root:
        root = random.choice(list(tree))
        with open(os.path.join(folder, 'root_cot.txt'), 'w+') as f:
            f.write(f"Motif {root} was chosen as the root.")
    else:
        while True:
            if tries == MAX_TRIES:
                logger.error(f"{folder} -- choose root {folder} exceeded max tries")
                sys.exit(1)
            root, cot = llm_choose_root(path, prompt_2_path, folder, text_only=args.text_only)
            if root in tree:
                break
            else:
                tries += 1
        with open(os.path.join(folder, 'root_cot.txt'), 'w+') as f:
            f.write(cot)
    if args.ablate_tree:        
        tree = nx.maximum_spanning_tree(tree)
    else:
        tree = llm_break_cycles(tree, mol, root, prompt_3_path, folder, scheme=args.scheme, text_only=args.text_only)
    
    if VISUALIZE:
        try:
            cot = llm_describe_tree(tree, mol, root, prompt_4_path, folder)
            with open(os.path.join(folder, 'tree_cot.txt'), 'w+') as f:
                f.write(cot)               
        except:
            logger.error(f"{folder} failed to produce tree description")     
            sys.exit(1)
    # tree = nx.maximum_spanning_tree(tree)

    # pos = hierarchy_pos(tree, root)

    # cpr.learn_production_rules(cg, tree, root)
    T = convert_to_node_set_tree(tree)
    set_root = frozenset(tree.nodes[root]["nodes"])
    rules = cpr.learn_production_rules(cg, T, set_root)
    # except:
    #     logger.error(f"cannot learn production rules for {smiles}")
    #     sys.exit(1)
    labels_lookup = {n: frozenset(tree.nodes[n]["nodes"]) for n in tree}
    inv_labels_lookup = dict(zip(labels_lookup.values(), labels_lookup.keys()))
    assert len(inv_labels_lookup) == len(labels_lookup)
    for symbol in rules:
        for r in rules[symbol]:
            node_set = frozenset(rules[symbol][r][1])
            assert node_set in inv_labels_lookup
            n = inv_labels_lookup[node_set]
            tree.nodes[n]["symbol"] = symbol
            tree.nodes[n]["rule"] = r
    # if VISUALIZE:
    fig, ax = plt.subplots()
    pos = hierarchy_pos(tree, root)
    nx.draw_networkx(tree, pos, ax=ax, with_labels=True, labels=labels_lookup)
    fig.savefig(os.path.join(folder, "tree.png"))
    logger.info(os.path.abspath(os.path.join(folder, "tree.png")))
    for n in tree:
        tree.nodes[n]['atoms'] = []
    order = []
    visited = {}
    def dfs(tree, cur, order, visited):
        order.append(cur)
        visited[cur] = True
        for nei in tree[cur]:
            if nei not in visited:
                dfs(tree, nei, order, visited)
    dfs(T, set_root, order, visited)
    for a in mol.GetAtoms():            
        b_idxes = set([b.GetIdx() for b in a.GetBonds()])
        for n in order:
            cond = len(b_idxes-n) == 0
            if cond:
                tree.nodes[inv_labels_lookup[n]]['atoms'].append(a.GetIdx())
                break
    if len(sum([tree.nodes[n]['atoms'] for n in tree], [])) != mol.GetNumAtoms():
        logger.error(f"{smiles} failed")
        raise ValueError(f"{smiles} failed")
    for n in tree:
        for attr in tree.nodes[n]        :
            tree.nodes[n][attr] = str(tree.nodes[n][attr])
    tree_path = os.path.join(folder, "tree.graphml")
    tree.graph['root'] = root
    nx.write_graphml(tree, tree_path)
    plt.close(fig)
    # test_rhs = list(rules['(S)'])[0]
    # logger.info(test_rhs)
    # matches = re.findall('(\((?:\d+,)*\d+:(?:N)\))', test_rhs)

    # i = 0
    # logger.info(matches[i])
    # grps = re.match('\(((?:\d+,)*\d+):(N)\)', matches[i])
    # nodes_idx_str, symbol = grps.groups()
    # nodes_idx = list(map(int, nodes_idx_str.split(',')))
    # bonds = [nodes[ind] for ind in nodes_idx]
    # __extract_subgraph(mol, GetAtomsFromBonds(mol, bonds))[0]
    g = Grammar(mol, rules)    
    g.folder = folder
    # g.VisRuleAlt('(a)',2)
    # rhs_mol = g.RHSMol('(a)',2)
    # logger.info(Grammar.atom_lookup(rhs_mol))
    # rhs_mol.GetBondBetweenAtoms(1,2).GetIdx()
    if VISUALIZE:
        fig = g.VisAllRules()
        for ri, r in enumerate(g.hrg.rules):
            r.rhs.visualize(os.path.join(folder, f"rule_{ri}.png"))
        fig.set_facecolor("white")
        fig.savefig(f"{folder}/rules.png")
        plt.close(fig)
    logger.info(f"{folder} -- finish learning grammar rules for {smiles}")
    return g, tree


def grammar_inference(G, trees):
    counts = [0 for _ in range(len(G.hrg.rules))]
    bad_smis = {'Cc1c(-c2c3nc4c(nc3c(-c3c(C)cc(-c5ccc6-c7ccccc7C(C)(C)c6c5)s3)s2)-c2cccc3cccc-4c23)scc1':'Cc1c(-c2c3nc4c(nc3c(-c3c(C)cc(-c5ccc6-c7ccccc7C(C)(C)c6c5)s3)s2)-c2cccc3c2c-4ccc3)scc1',
                'Cc1c(-c2c(C)cc(-c3cc(C)c(-c4c5nc6c(nc5c(-c5c(C)ccs5)s4)-c4cccc5cccc-6c45)s3)s2)scc1': 'Cc1c(-c2c(C)cc(-c3cc(C)c(-c4c5nc6c(nc5c(-c5c(C)ccs5)s4)-c4cccc5c4c-6ccc5)s3)s2)scc1',
                'O=[N+]([O-])c1ccc2CCc3cccc1c32': 'O=[N+]([O-])c1ccc2CCc3c2c1ccc3',                'COc1ccc([N+](=O)[O-])cc1N=Nc1c2ccccc2cc(C(=O)Nc2cccc([N+](=O)[O-])c2)c1O': 'COc1ccc([N+](=O)[O-])cc1N=Nc1c(O)c(C(=O)Nc2cccc([N+](=O)[O-])c2)cc2ccccc12'
                }
    for smi in trees:
        for tree in trees[smi]:
            for n in tree:
                symbol = tree.nodes[n]["symbol"]
                rule_str = tree.nodes[n]["rule"]
                if smi not in G.rule_idx_lookup:
                    smi = bad_smis[smi]
                rule_idx = G.rule_idx_lookup[smi][symbol][rule_str]                
                counts[rule_idx] += 1
    G.hrg.set_counts(counts)


def learn_grammar(smiles_or_list, args):
    logger = create_logger(
        "global_logger",
        f"{wd}/data/{METHOD}_{DATASET}_{GRAMMAR}-{args.mol_dataset}-{args.seed}.log",
    )
    
    if args.grammar_ckpt and os.path.exists(args.grammar_ckpt):
        G, trees = pickle.load(open(args.grammar_ckpt, 'rb'))
        exist_smiles = set([Chem.CanonSmiles(s) for s in G.mol_lookup])
        assert len(exist_smiles) == len(trees)
        smiles_or_list = list(filter(lambda x: Chem.CanonSmiles(x) not in exist_smiles, smiles_or_list))
    else:
        G, trees = None, None
    # for k, v in set_global_args(args).items():
    #     globals()[k] = v
    idx = 1
    while True:
        path = f"data/{args.mol_dataset}/api_mol_hg_{idx}.txt" 
        if os.path.exists(path):
            globals()[f'prompt_{idx}_path'] = path
            idx += 1
        else:
            break  

    def single_thread_execute(func, args, G=None, trees=None):
        if trees is None:
            trees = []
        for arg in args:
            g, tree = func(*arg)
            trees.append(tree)
            if G is None:
                G = g
            else:
                G = G.combine(g)
            if args[0][1].grammar_ckpt:
                if len(trees) > 1:
                    assert len(trees) == len(G.mol_lookup)
                pickle.dump(
                (G, trees),
                open(
                    args[0][1].grammar_ckpt, "wb+"
                ),
                )
                logger.info(f"dumped checkpoint: {len(trees)}")             
        return G, trees

    def multi_thread_execute(func, args, G=None, trees=None):
        if trees is None:
            trees = []       
        import threading 
        lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(NUM_THREADS) as executor:
            # Submit tasks to the thread pool
            futures = [
                executor.submit(func, *arg)
                for arg in tqdm(args, desc="submitting generation tasks")
            ]
            for future in concurrent.futures.as_completed(futures):
                with lock:
                    g, tree = future.result()
                    logger.info(f"{g.folder} completed")
                    trees.append(tree)
                    if G is None:
                        G = g
                    else:
                        G = G.combine(g)            
                    logger.info(f"{g.folder} combined")        
                    if args[0][1].grammar_ckpt:
                        if len(trees) > 1:
                            if len(trees) != len(G.mol_lookup):
                                breakpoint()
                            assert len(trees) == len(G.mol_lookup)
                        pickle.dump(
                        (G, trees),
                        open(
                            args[0][1].grammar_ckpt, "wb+"
                        ),
                        )
                        logger.info(f"dumped checkpoint: {len(trees)}")
        return G, trees

    if isinstance(smiles_or_list, list):
        task_args = [(smiles, args) for smiles in smiles_or_list]
        logger.info(f"===BEGIN LEARN GRAMMAR for {len(smiles_or_list)} MOLECULES===")
        if NUM_THREADS == 1:
            G, trees = single_thread_execute(_learn_grammar, task_args, G, trees)
        else:
            G, trees = multi_thread_execute(_learn_grammar, task_args, G, trees)
        # probabilistic grammar inference => populate counts
        breakpoint()
        if len(trees) > 1:
            grammar_inference(G, {smi: [tree] for (smi, tree) in zip(G.mol_lookup, trees)})
        pickle.dump(
            G,
            open(
                os.path.join(IMG_DIR, f"grammar-{args.mol_dataset}-{args.seed}.pkl"), "wb+"
            ),
        )
        pickle.dump(
            trees,
            open(os.path.join(IMG_DIR, f"trees-{args.mol_dataset}-{args.seed}.pkl"), "wb+"),
        )
        return G, trees
    else:
        logger.info(f"===BEGIN LEARN GRAMMAR for {smiles_or_list}===")
        return _learn_grammar(smiles_or_list, args)
