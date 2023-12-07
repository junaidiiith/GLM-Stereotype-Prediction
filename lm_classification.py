import transformers
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data import Dataset
import math
from tqdm.auto import tqdm
import torch
import math

import data_utils

model2key = {
    'bert-base-uncased': ['input_ids', 'token_type_ids', 'attention_mask'], 
    'distilbert-base-uncased-finetuned-sst-2-english': ['input_ids', 'attention_mask'],
    'YituTech/conv-bert-base': ['input_ids', 'token_type_ids', 'attention_mask'], 
}

def get_triples(graphs, distance=1, train=True, only_name=False):
    triples = list()
    for g, _ in tqdm(graphs):
        triples += get_graph_triples(g, distance=distance, train=train, only_name=only_name)
    return triples


def get_graph_triples(g, distance=1, train=True, only_name=False):
    relevant_nodes = [node for node in g.nodes if 'masked' in g.nodes[node] and g.nodes[node]['masked'] != train]
    node_strings = [data_utils.get_node_str(g, node, distance) for node in relevant_nodes]
    node_triples = list()
    for node, node_str in zip(relevant_nodes, node_strings):
        name = g.nodes[node]['name']
        node_type = g.nodes[node]['type']
        if node_str == "":
            node_str = name
        label_str = g.nodes[node]['stereotype']
        # prompt_str = f"{node_type}"
        prompt_str = f"{node_type} {name}: {node_str}" if not only_name else f"{name}"
        node_triples.append((prompt_str, label_str))
    return node_triples


def get_triples_dataset(triples, label_encoder, tokenizer, max_length):
    max_length = max_length if max_length < 512 else 512
    inputs, labels = [i[0] for i in triples], [label_encoder[i[1]] for i in triples]
    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    dataset = EncodingsDataset(input_encodings, labels)
    return dataset


# Create a custom dataset
class EncodingsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        output = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
        # if 'token_type_ids' in self.encodings:
        #     output['token_type_ids'] = self.encodings['token_type_ids'][idx]
        return output

def get_stats(trainer, dataset):
    eval_results = trainer.evaluate(dataset)
    predictions = get_predictions(trainer.model, dataset)
    return {
        'loss': eval_results['eval_loss'], 
        'perplexity': math.exp(eval_results['eval_loss']), 
        'accuracy': eval_results['eval_accuracy'],
        'predictions': predictions
    }


def get_eval_stats(eval_result):
    stats = {
        'loss': eval_result['eval_loss'], 
        'perplexity': math.exp(eval_result['eval_loss']), 
        'accuracy': eval_result['eval_accuracy'],
    }
    # stats = {
    #     'loss': 0.4,
    #     'perplexity': 0.4,
    #     'accuracy': 0.4,
    # }
    return stats


def get_predictions(model, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    predictions, actuals = list(), list()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(torch.long).to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions += outputs.logits.argmax(-1).tolist()
            actuals += batch['labels'].tolist()

    pred_dist = data_utils.get_predictions_distribution(predictions, actuals)
    return pred_dist


def train_lm(seen_graphs, unseen_graphs, label_encoder, args):
    results = {
        'model_name': args.model_name, 
        'distance': args.distance, 
        'seed': args.seed, 
        'use_rel_stereotypes': args.use_rel_stereotypes, 
        'use_stereotypes': args.use_stereotypes
    }
    model_name = args.model_name
    train_triples_seen = get_triples(seen_graphs, distance=args.distance, train=True, only_name=args.only_name)
    test_triples_seen = get_triples(seen_graphs, distance=args.distance, train=False, only_name=args.only_name)

    train_triples_unseen = get_triples(unseen_graphs, distance=args.distance, train=True, only_name=args.only_name)
    test_triples_unseen = get_triples(unseen_graphs, distance=args.distance, train=False, only_name=args.only_name)
    print(len(train_triples_seen), len(test_triples_seen))
    print(len(train_triples_unseen), len(test_triples_unseen))

    print(train_triples_seen[:2])
    print(test_triples_seen[:2])
    print(train_triples_unseen[:2])
    print(test_triples_unseen[:2])

    # input("Press enter to continue...")
    tokenizer = data_utils.get_tokenizer(model_name)

    # mx_len = 128
    mx_len, _, _ = data_utils.get_encoding_size(train_triples_seen, tokenizer)
    train_dataset_seen = get_triples_dataset(train_triples_seen, label_encoder, tokenizer, max_length=mx_len)

    mx_len, _, _ = data_utils.get_encoding_size(test_triples_seen, tokenizer)
    test_dataset_seen = get_triples_dataset(test_triples_seen, label_encoder, tokenizer, max_length=mx_len)

    mx_len, _, _ = data_utils.get_encoding_size(train_triples_unseen, tokenizer)
    train_dataset_unseen = get_triples_dataset(train_triples_unseen, label_encoder, tokenizer, max_length=mx_len)

    mx_len, _, _ = data_utils.get_encoding_size(test_triples_unseen, tokenizer)
    test_dataset_unseen = get_triples_dataset(test_triples_unseen, label_encoder, tokenizer, max_length=mx_len)

    print("Seen train dataset size:", len(train_dataset_seen))
    print("Seen test dataset size:", len(test_dataset_seen))
    print("Unseen train dataset size:", len(train_dataset_unseen))
    print("Unseen test dataset size:", len(test_dataset_unseen))
    
    batch_size = args.lm_batch_size
    # Show the training loss with every epoch
    logging_steps = len(train_dataset_seen) // batch_size
    print(f"Using model...{model_name}")
    model = data_utils.get_classification_model(model_name, len(label_encoder), tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    print("Finetuning model...")
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        logging_steps=logging_steps,
        num_train_epochs=args.num_epochs_lm,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_seen,
        eval_dataset=test_dataset_seen,
        tokenizer=tokenizer,
        compute_metrics=data_utils.compute_metrics,
    )
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, transformers.integrations.NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)

    eval_results = trainer.evaluate()
    print(f">>> Accuracy (before training): {eval_results['eval_accuracy']:.2f}")
    results['seen_train (before)'] = results['seen_train (after)'] = get_stats(trainer, test_dataset_seen)

    trainer.eval_dataset = train_dataset_unseen
    eval_results = trainer.evaluate()
    print(f">>> Accuracy (before training): {eval_results['eval_accuracy']:.2f}")
    results['unseen_train (before)'] = results['seen_train (after)'] = get_stats(trainer, train_dataset_unseen)

    trainer.eval_dataset = test_dataset_unseen
    eval_results = trainer.evaluate()
    print(f">>> Accuracy (before training): {eval_results['eval_accuracy']:.2f}")

    results['unseen_test (before)'] = results['seen_train (after)'] = get_stats(trainer, test_dataset_unseen)
    
    print("Begin training...")

    trainer.train()
    trainer.save_model()
    eval_results = trainer.evaluate()
    print(f">>> Accuracy (after training): {eval_results['eval_accuracy']:.2f}")
    results['seen_train (after)'] = get_eval_stats(eval_results)
    # results['seen_train (after)'] = get_stats(trainer, test_dataset_seen)


    trainer.eval_dataset = train_dataset_unseen
    eval_results = trainer.evaluate()
    results['unseen_train (after)'] = get_eval_stats(eval_results)
    print(f">>> Accuracy (after training): {eval_results['eval_accuracy']:.2f}")

    trainer.eval_dataset = test_dataset_unseen
    eval_results = trainer.evaluate()
    results['unseen_test (after)'] = get_eval_stats(eval_results)
    
    return results


# def main():
#     from parameters import parse_args
#     args = parse_args()
#     data_utils.set_seed(args.seed)
#     seen_graphs, unseen_graphs, label_encoder = data_utils.get_graphs_data(args)
#     print(train_lm(seen_graphs, unseen_graphs, label_encoder, args))

# main()
