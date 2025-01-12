Installation

1. git clone [this repo]
2. git submodule init ; git submodule update
3. conda env create -f dagnn_env.yml
4. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Grammar Induction
1. Run python main.py

Configure a new dataset

1. Make a new yml in src/config/, in same format as the other examples.
2. Add an elif block in def load_data() in main.py
3. Write a load_{your dataset} function in src/examples/test_graphs.py