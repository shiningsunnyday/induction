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


@retry_with_exponential_backoff
def create_chat_completion(**kwargs):
    hash_ = hash_dict(kwargs)
    cache_path = os.path.join(IMG_DIR, f"{hash_}.txt")
    # if os.path.exists(cache_path):
    if False:
        res = open(cache_path).read()
    else:
        completion = openai.ChatCompletion.create(**kwargs)
        res = completion.choices[0].message.content
        with open(cache_path, "w+") as f:
            f.write(res)
    return res


def llm_call(img_paths, prompt_path, optional_prompt=None):
    """
    This function uses prompt read from prompt_path and a list of img content.
    Parameters:
        img_paths: list of paths to img files
        prompt_path: a .txt file path
        optional_prompt: lambda function with single arg
        optional text prompt to process the output of the response
    Output:
        Response of call
    """
    logger = logging.getLogger("global_logger")
    # settings = {
    #     "temperature": 0,
    #     "seed": 42,
    #     "top_p": 0,
    #     "n": 1
    # } # try to make deterministic
    settings = {}
    base64_images = prepare_images(img_paths)
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
        **settings,
    )
    logger.info("===PROMPT===")
    logger.info(prompt)
    logger.info("===RESPONSE===")
    logger.info(res + "\n")
    if optional_prompt:
        res = create_chat_completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": optional_prompt(res)}],
                }
            ],
            **settings,
        )
        logger.info("===PROMPT===")
        logger.info(optional_prompt(res))
        logger.info("===RESPONSE===")
        logger.info(res + "\n")
    return res


def llm_choose_edit(img_path, prompt_path):
    post_prompt = (
        lambda res: f"I want you to perform a simple data post-processing step of the following response:\n{res}\n The input is a response from another language agent. It may or may not contain an answer in the form of a single pair. If it does, output the pair in x,y format and NOTHING ELSE. Don't include explanations or superlatives. Just output the answer. If it doesn't contain an answer in the form of a pair, output the single word NONE."
    )
    return llm_call([img_path], prompt_path, post_prompt)


def llm_edit_cliques(cg, mol, prompt_path):
    d = get_next_version(IMG_DIR)
    dir_name = os.path.join(IMG_DIR, f"{d}")
    os.makedirs(dir_name)
    while True:
        i = get_next_version(dir_name, dir=False)
        path = os.path.join(dir_name, f"{i}.png")
        cliques = clique_drawing(cg, mol, path)
        pair = llm_choose_edit(path, prompt_path)
        match = re.match(f"(\d+),(\d+)", pair)
        if match:
            e1 = int(match.groups()[0])
            e2 = int(match.groups()[1])
            if max(e1, e2) >= len(cliques):
                break
        else:
            break
        cq = cliques[e1] + cliques[e2]
        cg.add_edges_from(product(cq, cq))
        cg.remove_edges_from(nx.selfloop_edges(cg))
    return cg, path


def llm_choose_root(img_path, prompt_path):
    post_prompt = (
        lambda res: f"I want you to perform a simple data post-processing step of the following response:\n{res}\n The input is a response from another language agent. It may or may not contain an answer in the form of a single integer. If it does, output the integer and NOTHING ELSE. Don't include explanations or superlatives. Just output the answer. If it doesn't contain an answer in the form of a pair, output the single word NONE."
    )
    root = llm_call([img_path], prompt_path, post_prompt)
    match = re.match("^\d+$", root)
    if match:
        return int(root)
    else:
        return 0


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


def llm_break_edge(img_path, prompt_path):
    post_prompt = (
        lambda res: f"I want you to perform a simple data post-processing step of the following response:\n{res}\n The input is a response from another language agent. It may or may not contain an answer in the form of a single integer. If it does, output the integer and NOTHING ELSE. Don't include explanations or superlatives. Just output the answer. If it doesn't contain an answer in the form of a pair, output the single word NONE."
    )
    return llm_call([img_path], prompt_path, post_prompt)


def llm_break_cycles(tree, mol, root, prompt_path):
    d = get_next_version(IMG_DIR)
    dir_name = os.path.join(IMG_DIR, f"{d}")
    os.makedirs(dir_name)
    while not nx.is_tree(tree):
        i = get_next_version(dir_name, dir=False)
        path = os.path.join(dir_name, f"{i}.png")
        cyc = nx.find_cycle(tree, root)
        if cyc:
            path = draw_cycle(cyc, tree, mol, path)
            e = llm_break_edge(path, prompt_path)
            match = re.match("^\d+$", e)
            if match:
                e = int(e)
                if e >= len(cyc):
                    continue
                e1, e2 = cyc[e]
                tree.remove_edge(e1, e2)
        else:
            break
    return tree


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


def _learn_grammar(smiles):
    logger = logging.getLogger("global_logger")
    logger.info(f"begin learning grammar rules for {smiles}")
    draw_smiles(smiles, path=os.path.join(IMG_DIR, "smiles.png"))
    mol, cg = chordal_mol_graph(smiles)
    cg, path = llm_edit_cliques(cg, mol, prompt_1_path)
    tree = init_tree(cg)
    logger.info(nx.is_tree(tree))
    while True:
        root = llm_choose_root(path, prompt_2_path)
        if root in tree:
            break
    tree = llm_break_cycles(tree, mol, root, prompt_3_path)
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
        fig.savefig(os.path.join(IMG_DIR, "tree.png"))
        logger.info(os.path.abspath(os.path.join(IMG_DIR, "tree.png")))
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
    # g.VisRuleAlt('(a)',2)
    # rhs_mol = g.RHSMol('(a)',2)
    # logger.info(Grammar.atom_lookup(rhs_mol))
    # rhs_mol.GetBondBetweenAtoms(1,2).GetIdx()
    if VISUALIZE:
        fig = g.VisAllRules()
        for ri, r in enumerate(g.hrg.rules):
            r.rhs.visualize(os.path.join(IMG_DIR, f"rule_{ri}.png"))
        fig.set_facecolor("white")
        fig.savefig(f"{IMG_DIR}/rules.png")
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

    def single_thread_execute(func, args):
        G = None
        trees = []
        for a in args:
            g, tree = func(a)
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
                executor.submit(func, arg)
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
        logger.info(f"===BEGIN LEARN GRAMMAR for {len(smiles_or_list)} MOLECULES===")
        if NUM_THREADS == 1:
            G, trees = single_thread_execute(_learn_grammar, smiles_or_list)
        else:
            G, trees = multi_thread_execute(_learn_grammar, smiles_or_list)
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
        return _learn_grammar(smiles_or_list)
