from tqdm.auto import tqdm
import itertools
import json
from lm_classification import train_lm
from mlm_classification import train_mlm
from gnn_train import train_gnn
import data_utils
from parameters import parse_args


def main():
    args = parse_args()
    data_utils.set_seed(args.seed)

    results_dir = args.results_dir

    distances = [(0, 128), (1, 64), (2, 32), (3, 16)]
    # distances = [(2, 32)]
    # model_name = args.model_name
    # language_models = ['YituTech/conv-bert-base', 'distilbert-base-uncased-finetuned-sst-2-english']
    language_models = ['bert-base-uncased']
    exclude_limit = [100]
    config_results = dict()

    configs = list(itertools.product(distances, exclude_limit, language_models))
    for distance, exclude_limit, model_name in tqdm(configs, desc=f'Running config'):
        args.model_name = model_name
        args.distance, args.lm_batch_size = distance
        args.exclude_limit = exclude_limit
        suffix = f"-s_{args.seed}_d_{args.distance}_el_{args.exclude_limit}"
        config_string = f'model_{model_name.replace("/", "_")}_seed_{args.seed}_distance_{args.distance}_exclude_limit_{args.exclude_limit}'
        kfold_results = dict()

        args.only_name = True if distance[0] == 0 else False

        for i, (seen_graphs, unseen_graphs, label_encoder) in enumerate(data_utils.get_graphs_data_kfold(args)):
            print(f'Running fold {i}...')
            kfold_results['label encoder'] = label_encoder
            results = dict()
            results['seen info'] = data_utils.get_graphs_info(seen_graphs)
            results['unseen info'] = data_utils.get_graphs_info(unseen_graphs)
            
            args.model_name = model_name
            args.out_dir = f'trained_models/{model_name}' + '-mlm' + suffix
            mlm_results = train_mlm(seen_graphs, unseen_graphs, args)
            
            results['mlm-vanilla'] = mlm_results
            
            args.out_dir = f'trained_models/{model_name}' + '-lm-seq' + suffix
            lm_results = train_lm(seen_graphs, unseen_graphs, label_encoder, args)
            results['lm-vanilla'] = lm_results
            
            args.model_name = f'trained_models/{model_name}' + '-mlm' + suffix
            args.out_dir = f'trained_models/{model_name}' + '-mlm-lm-seq' + suffix
            lm_results = train_lm(seen_graphs, unseen_graphs, label_encoder, args)
            results['lm-mlm'] = lm_results

            print("Training GNN model with BERT and Word2Vec embeddings...")
            model_names = [
                f'{model_name}',
                f'trained_models/{model_name}' + '-mlm' + suffix,
                f'trained_models/{model_name}' + '-lm-seq' + suffix,
                f'trained_models/{model_name}' + '-mlm-lm-seq' + suffix,
                f'sgram-mde',
                f'onto',
                f'small'
            ]
            for m_name in tqdm(model_names, desc=f'Running GNN config'):
                print(f"Training GNN model with {m_name} embeddings...")
                args.model_name = m_name
                if 'bert' in m_name:
                    gnn_lm_results = train_gnn(seen_graphs, unseen_graphs, label_encoder, args)
                    results[m_name] = gnn_lm_results
                else:
                    args.use_context = False
                    gnn_w2v_results_without_context = train_gnn(seen_graphs, unseen_graphs, label_encoder, args)
                    results[f'{m_name}-without-context'] = gnn_w2v_results_without_context
                    args.use_context = True
                    gnn_w2v_results_with_context = train_gnn(seen_graphs, unseen_graphs, label_encoder, args)
                    results[f'{m_name}-with-context'] = gnn_w2v_results_with_context
            

            kfold_results[f'fold-{i}'] = results
            
            
            with open(f'{results_dir}/kfold_results_{config_string}.json', 'w') as f:
                json.dump(kfold_results, f, indent=4)
            
            
        config_results[config_string] = kfold_results
        with open(f'{results_dir}/config_results.json', 'w') as f:
            json.dump(config_results, f, indent=4)


if __name__ == "__main__":
    main()