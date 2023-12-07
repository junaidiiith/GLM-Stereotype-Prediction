from torch.utils.data import Dataset
import itertools
from trainers import GraphTrainer
import dgl
from dgl.dataloading import GraphDataLoader
from parameters import parse_args
from torch.utils.data import Dataset
from parameters import parse_args
from tqdm.auto import tqdm
from gnn_layers import GNNModel
import data_utils


collate_keys = [
    'raw_embeddings', 
    'wl_embeddings', 
    'hop_embeddings', 
    'int_embeddings', 
    'train_mask',
    'val_mask',
    'X', 'y'
]

class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def train_gnn(seen_graphs, unseen_graphs, label_encoder, args):
    model_name, distance, use_context = args.model_name, args.distance, args.use_context
    seen_dgl_graphs = data_utils.create_dgl_graphs(seen_graphs, label_encoder, model_name, distance, use_context)
    unseen_dgl_graphs = data_utils.create_dgl_graphs(unseen_graphs, label_encoder, model_name, distance, use_context)
    gnn_config_results = dict()
    seen_dataset = GraphDataset(seen_dgl_graphs)
    unseen_dataset = GraphDataset(unseen_dgl_graphs)
    
    def collate_fn(graphs):
        batched_graph = dgl.batch(graphs)
        return batched_graph

    train_dataloader = GraphDataLoader(seen_dataset, batch_size=args.graph_batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = GraphDataLoader(unseen_dataset, batch_size=args.graph_batch_size, shuffle=False, collate_fn=collate_fn)

    args.num_training_steps = args.num_epochs * len(train_dataloader)
    embed_dim = seen_dataset[0].ndata['feat'].shape[1]

    # models = [('GINConv', False), ('GCNConv', False), ('GATConv', True), ('SAGEConv', False)]
    models = [('SAGEConv', False)]
    # residual = [True, False]
    residual = [True]
    # num_heads = [2, 4, None]
    num_heads = [None]
    num_layers = [2, 3]
    gnn_config_results = dict()
    print("Training GNN configs...")
    all_combinations = list(itertools.product(models, residual, num_heads, num_layers))
    all_combinations = [c for c in all_combinations if (c[0][1] and c[2] is not None) or (not c[0][1] and c[2] is None)]
    
    for model, residual, num_heads, num_layers in tqdm(all_combinations, desc=f'Running GNN config'):
    # for model, residual, num_heads, num_layers in all_combinations:
        model_name, _ = model
        gnn_model = GNNModel(
            model_name=model_name, 
            input_dim=embed_dim, 
            hidden_dim=128, 
            out_dim=len(label_encoder), 
            num_layers=num_layers, 
            num_heads=num_heads, 
            residual=residual
        )
        gnn_results = dict()
        gnn_trainer = GraphTrainer(gnn_model, args)
        print("Result before training: ")
        gnn_trainer.validate(test_dataloader)
        print("Training GNN model...")
        gnn_results['train'] = gnn_trainer.run_epochs(train_dataloader, args.num_epochs)
        print("Results after training: ")
        gnn_results['test'] = gnn_trainer.validate(test_dataloader)
        config_str = f'{model_name}-residual-{residual}-num_heads-{num_heads}-num_layers-{num_layers}'
        gnn_config_results[config_str] = gnn_results
    
    print("Training GNN configs done.")
    return gnn_config_results


# def main():
#     args = parse_args()
#     seen_graphs, unseen_graphs, label_encoder = data_utils.get_graphs_data(args)

    # print("Training GNN model with BERT embeddings...")
    # print(train_gnn(seen_graphs, unseen_graphs, label_encoder, 'bert-base-uncased', args))

    # print("Training GNN model with word2vec embeddings...")
    # print(train_gnn(seen_graphs, unseen_graphs, label_encoder, 'sgram-mde', args))

    


# if __name__ == "__main__":
#     main()