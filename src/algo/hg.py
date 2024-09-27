from src.grammar.hg import *
from src.config import *
from src.draw.mol import *
import hashlib
import time
from filelock import FileLock


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
            return logprobs['content'][0]['top_logprobs']
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
    logger = logging.getLogger("global_logger")
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
    # settings = {
    #     "temperature": 0,
    #     "seed": 42,
    #     "top_p": 0,
    #     "n": 1
    # } # try to make deterministic
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
        if VERBOSE:
            logger.info("=====END LLM call=====")
        return res
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


def llm_edit_cliques(cg, mol, prompt_path, folder, scheme='zero'):
    # if folder is None:
    #     d = get_next_version(IMG_DIR)
    #     dir_name = os.path.join(IMG_DIR, f"{d}")        
    # else:
    dir_name = os.path.join(folder, 'cliques/')
    os.makedirs(dir_name, exist_ok=True)
    cots = []
    while True:
        i = get_next_version(dir_name, dir=False)
        path = os.path.join(dir_name, f"{i}.png")
        cliques = clique_drawing(cg, mol, path, scheme=scheme)
        path_indv = os.path.join(dir_name, f"{i}_indv.png")
        clique_drawing(cg, mol, path_indv, scheme=scheme, isolate=True)        
        isolate_cot = llm_describe_cliques(cliques, [path_indv], prompt_6_path)        
        combined_prompt = ''.join(open(prompt_path).readlines())
        combined_prompt = combined_prompt.replace("<optional>", isolate_cot)
        pair, cot = llm_choose_edit([path, path_indv], None, prompt=combined_prompt)
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
    while str(num) in res:
        num += 1
    return num


def llm_describe_cliques(cliques, paths, prompt_path):
    numbering_str = ', '.join(map(str, range(len(cliques))))
    post_prompt = (
        lambda res: f"I want you to do a simple check of the following response:\n{res}\n The input is a response from another language agent. I want you to sanity check if it contains descriptions for PRECISELY {len(cliques)} motifs, numbered from {numbering_str}. If there is an issue with the numbering, or the response is missing some motifs, the response fails the sanity check. Output YES if the response passes the sanity check, and NO otherwise."
    )
    prompt = ''.join(open(prompt_path).readlines())
    if len(cliques) > 1:
        prompt = prompt.replace("<optional>", "I will highlight for you some of the distinctive substructures of a molecule. They are numbered from 0.")
    else:
        prompt = prompt.replace("<optional>", "I will highlight for you one distinctive substructure of a molecule.")        
    tries = 0
    while True:
        if tries == MAX_TRIES:
            breakpoint()
        ans_heads, res = llm_call(paths, None, optional_prompts=[post_prompt], prompt=prompt)
        ans = ans_heads[0]
        if ans[:3] == "YES" and sanity_check_num_cliques(res) == len(cliques):
            break                
        tries += 1
    return res


def llm_choose_root(img_path, prompt_path, folder, scheme='zero'):
    post_prompt = (
        lambda res: f"I want you to perform a simple data post-processing step of the following response:\n{res}\n The input is a response from another language agent. It may or may not contain an answer in the form of a single integer. If it does, output the integer and NOTHING ELSE. Don't include explanations or superlatives. Just output the answer. If it doesn't contain an answer in the form of a pair, output the single word NONE."
    )
    motif_cot = ''.join(open(os.path.join(folder, 'motifs_cot.txt')).readlines())    
    combined_prompt = ''.join(open(prompt_path).readlines())    
    combined_prompt = combined_prompt.replace("<optional>", motif_cot)    
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


def llm_break_edge(img_path, prompt_path, prompt=None):
    post_prompt = (
        lambda res: f"I want you to perform a simple data post-processing step of the following response:\n{res}\n The input is a response from another language agent. It may contain an answer in the form of a single integer for the LEAST important interaction. If it does, output the integer and NOTHING ELSE. Don't include explanations or superlatives. Just output the answer. If it doesn't contain an answer in the form of a pair, output the single word NONE."
    )
    ans_heads, _ = llm_call([img_path], prompt_path, [post_prompt], prompt=prompt)
    return ans_heads[0]


def llm_break_cycles(tree, mol, root, prompt_path, folder, scheme='zero'):
    describe_post_prompt = (
        lambda res: f"I want you to perform a simple post-processing step of the following response:\n{res}\n The input is a response from another language agent. It describes motifs numbered from 0. I want you to rephrase each motif description by filling in X within the following sentence template: \nThis motif is X\n Be sure to condense the description and output a single PHRASE such that the sentence template is grammatically correct. Don't capitalize the first letter, since your answer should just be a phrase. Output your rephrasing starting for Motif 0, line-by-line. Output your rephrasing for each motif on a SEPARATE line, using only a new line for delimiting different motifs. DON'T output anything else."
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
            breakpoint()
        describe_cot = llm_call([], None, prompt=prompt)
        describes = [line for line in describe_cot.split('\n') if line]
        if len(describes) == len(tree):
            break
        tries += 1
    while not nx.is_tree(tree):
        i = get_next_version(dir_name, dir=False)
        path = os.path.join(dir_name, f"{i}.png")
        cyc = nx.find_cycle(tree, root)
        if cyc:
            path = draw_cycle(cyc, tree, mol, path)
            combined_prompt = ''.join(open(prompt_path).readlines())
            interaction_descrs = [f"Interaction {i} features {describes[c[0]]} and {describes[c[1]]}." for i, c in enumerate(cyc)]
            interaction_descr = '\n'.join(interaction_descrs)
            combined_prompt = combined_prompt.replace('<optional>', interaction_descr)
            e = llm_break_edge(path, prompt_path, prompt=combined_prompt)
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
    logger = logging.getLogger("global_logger")
    logger.info(f"begin learning grammar rules for {smiles}")
    folder = f"data/api_mol_hg/learn-{time.time()}/"
    os.makedirs(folder, exist_ok=True)
    draw_smiles(smiles, path=os.path.join(folder, "smiles.png"), label_bonds=False)
    with open(os.path.join(folder, "smiles.txt"), "w+") as f:
        f.write(f"{smiles}")        
    draw_smiles(smiles, path=os.path.join(folder, "smiles_labeled.png"), label_atoms=True, label_atom_idx=True, label_bonds=True)    
    mol, cg, base_cg = chordal_mol_graph(smiles)
    if VISUALIZE:
        base_motif_path = os.path.join(folder, "base_motifs.png")        
        clique_drawing(base_cg, mol, path=base_motif_path, scheme=args.scheme)
    cg, path, cots = llm_edit_cliques(cg, mol, prompt_1_path, folder=folder, scheme=args.scheme)
    for i, (_, clique_cot) in enumerate(cots):
        with open(os.path.join(folder, f'clique_{i}_cot.txt'), 'w+') as f:
            f.write(clique_cot)
    tree = init_tree(cg)
    if VERBOSE:
        logger.info(f"Initialized tree is tree? {nx.is_tree(tree)}")
    motif_path = os.path.join(folder, "motifs.png")
    clique_drawing(cg, mol, path=motif_path, isolate=True, scheme=args.scheme)

    with open(os.path.join(folder, 'motifs_cot.txt'), 'w+') as f:
        f.write(cots[-1][0])
    tries = 0
    while True:
        if tries == MAX_TRIES:
            breakpoint()
        root, cot = llm_choose_root(path, prompt_2_path, folder)
        if root in tree:
            break
        else:
            tries += 1
    with open(os.path.join(folder, 'root_cot.txt'), 'w+') as f:
        f.write(cot)
    tree = llm_break_cycles(tree, mol, root, prompt_3_path, folder, scheme=args.scheme)
    try:
        cot = llm_describe_tree(tree, mol, root, prompt_4_path, folder)
    except:
        logger.error(f"{smiles} failed to produce tree description")
    with open(os.path.join(folder, 'tree_cot.txt'), 'w+') as f:
        f.write(cot)        
    # tree = nx.maximum_spanning_tree(tree)
    # while not nx.is_tree(tree):
    # nx.is_tree(tree)

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
    if VISUALIZE:
        fig, ax = plt.subplots()
        pos = hierarchy_pos(tree, root)
        nx.draw_networkx(tree, pos, ax=ax, with_labels=True, labels=labels_lookup)
        fig.savefig(os.path.join(folder, "tree.png"))
        logger.info(os.path.abspath(os.path.join(folder, "tree.png")))
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
    logger.info(f"finish learning grammar rules for {smiles}")
    return g, tree


def grammar_inference(G, trees):
    counts = [0 for _ in range(len(G.hrg.rules))]
    for smi, tree in zip(G.mol_lookup, trees):
        for n in tree:
            symbol = tree.nodes[n]["symbol"]
            rule_str = tree.nodes[n]["rule"]
            rule_idx = G.rule_idx_lookup[smi][symbol][rule_str]
            counts[rule_idx] += 1
    G.hrg.set_counts(counts)


def learn_grammar(smiles_or_list, args):
    logger = create_logger(
        "global_logger",
        f"{wd}/data/{METHOD}_{DATASET}_{GRAMMAR}-{args.dataset}-{args.seed}.log",
    )

    # for k, v in set_global_args(args).items():
    #     globals()[k] = v
    idx = 1
    while True:
        path = f"data/{args.dataset}/api_mol_hg_{idx}.txt" 
        if os.path.exists(path):
            globals()[f'prompt_{idx}_path'] = path
            idx += 1
        else:
            break  
    def single_thread_execute(func, args):
        G = None
        trees = []
        for arg in args:
            g, tree = func(*arg)
            trees.append(tree)
            if G is None:
                G = g
            else:
                G = G.combine(g)
        return G, trees

    def multi_thread_execute(func, args):
        G = None
        trees = []
        with concurrent.futures.ThreadPoolExecutor(NUM_THREADS) as executor:
            # Submit tasks to the thread pool
            futures = [
                executor.submit(func, *arg)
                for arg in tqdm(args, desc="submitting generation tasks")
            ]
            for future in concurrent.futures.as_completed(futures):
                g, tree = future.result()
                trees.append(tree)
                if G is None:
                    G = g
                else:
                    G = G.combine(g)
        return G, trees

    if isinstance(smiles_or_list, list):
        task_args = [(smiles, args) for smiles in smiles_or_list]
        logger.info(f"===BEGIN LEARN GRAMMAR for {len(smiles_or_list)} MOLECULES===")
        if NUM_THREADS == 1:
            G, trees = single_thread_execute(_learn_grammar, task_args)
        else:
            G, trees = multi_thread_execute(_learn_grammar, task_args)
        # probabilistic grammar inference => populate counts
        grammar_inference(G, trees)
        pickle.dump(
            G,
            open(
                os.path.join(IMG_DIR, f"grammar-{args.dataset}-{args.seed}.pkl"), "wb+"
            ),
        )
        pickle.dump(
            trees,
            open(os.path.join(IMG_DIR, f"trees-{args.dataset}-{args.seed}.pkl"), "wb+"),
        )
        return G, trees
    else:
        logger.info(f"===BEGIN LEARN GRAMMAR for {smiles_or_list}===")
        return _learn_grammar(smiles_or_list, args)
