import transformers
from transformers import TrainingArguments, pipeline
from transformers import Trainer
from torch.utils.data import Dataset
import math
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForMaskedLM
import math
import data_utils



HIDE_TOKEN = '[HIDE]'
PATHS = "[PATHS]"

def get_triples(graphs, distance=1, train=True):
    triples = list()
    for g, _ in tqdm(graphs, desc=f'Creating graph triples'):
        triples += get_graph_triples(g, distance=distance, train=train)
    return triples


def get_graph_triples(g, distance=1, train=True):
    relevant_nodes = [node for node in g.nodes if 'masked' in g.nodes[node] and g.nodes[node]['masked'] != train]
    node_strings = [data_utils.get_node_str(g, node, distance) for node in relevant_nodes]
    node_triples = list()
    for node, node_str in zip(relevant_nodes, node_strings):
        name = g.nodes[node]['name'] if g.nodes[node]['name'] != "Null" else " reference "
        node_type = g.nodes[node]['type']
        stereotype = g.nodes[node]['stereotype']
        prompt_str = f"{node_type} {name} {data_utils.STEREOTYPE} {HIDE_TOKEN}{stereotype}{HIDE_TOKEN} | {name}{PATHS}{node_str}"
        node_triples.append(prompt_str)
    return node_triples


def get_triples_dataset(triples, tokenizer, max_length):
    # print("Tokenizing triples vocab size", len(tokenizer))
    max_length = max_length if max_length < 512 else 512
    encodings = tokenizer(triples, truncation=True, padding=True, max_length=max_length)
    # print("Max token size before update: ", max(max(i) for i in encodings['input_ids']))
    HIDE_TOKEN_ID = tokenizer.convert_tokens_to_ids(HIDE_TOKEN)
    encodings = data_utils.update_encodings(encodings, HIDE_TOKEN_ID, tokenizer.mask_token_id, tokenizer.pad_token_id)
    dataset = EncodingsDataset(encodings)
    # print("Max token size after update: ", max(max(i['input_ids']) for i in dataset))
    return dataset


# Create a custom dataset
class EncodingsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings['labels'][idx],
        }
    
def train_mlm(seen_graphs, unseen_graphs, args):
    results = {
        'model_name': args.model_name, 
        'distance': args.distance, 
        'seed': args.seed, 
        'use_rel_stereotypes': args.use_rel_stereotypes, 
        'use_stereotypes': args.use_stereotypes
    }
    model_name = args.model_name
    train_triples_seen = get_triples(seen_graphs, distance=args.distance, train=True)
    test_triples_seen = get_triples(seen_graphs, distance=args.distance, train=False)

    train_triples_unseen = get_triples(unseen_graphs, distance=args.distance, train=True)
    test_triples_unseen = get_triples(unseen_graphs, distance=args.distance, train=False)
    print(len(train_triples_seen), len(test_triples_seen))
    print(len(train_triples_unseen), len(test_triples_unseen))

    print(train_triples_seen[:2])
    print(test_triples_seen[:2])
    print(train_triples_unseen[:2])
    print(test_triples_unseen[:2])

    # input("Press enter to continue...")

    total_mask = lambda dataset: sum(torch.where(i['input_ids'] == tokenizer.mask_token_id)[0].shape[0] for i in dataset)
    tokenizer = data_utils.get_tokenizer(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': [HIDE_TOKEN, data_utils.STEREOTYPE, PATHS]})

    print("Model name", model_name)
    print("Tokenizer vocab size", len(tokenizer))

    mx_len, _, _ = data_utils.get_encoding_size(train_triples_seen, tokenizer)
    train_dataset_seen = get_triples_dataset(train_triples_seen, tokenizer, max_length=mx_len)

    mx_len, _, _ = data_utils.get_encoding_size(test_triples_seen, tokenizer)
    test_dataset_seen = get_triples_dataset(test_triples_seen, tokenizer, max_length=mx_len)

    mx_len, _, _ = data_utils.get_encoding_size(train_triples_unseen, tokenizer)
    train_dataset_unseen = get_triples_dataset(train_triples_unseen, tokenizer, max_length=mx_len)

    mx_len, _, _ = data_utils.get_encoding_size(test_triples_unseen, tokenizer)
    test_dataset_unseen = get_triples_dataset(test_triples_unseen, tokenizer, max_length=mx_len)

    print("Seen train dataset size:", len(train_dataset_seen), "Total masked tokens", total_mask(train_dataset_seen))
    print("Seen test dataset size:", len(test_dataset_seen), "Total masked tokens", total_mask(test_dataset_seen))
    print("Unseen train dataset size:", len(train_dataset_unseen), "Total masked tokens", total_mask(train_dataset_unseen))
    print("Unseen test dataset size:", len(test_dataset_unseen), "Total masked tokens", total_mask(test_dataset_unseen))

    train_dataloader = torch.utils.data.DataLoader(train_dataset_seen, batch_size=args.lm_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset_seen, batch_size=args.lm_batch_size, shuffle=True)
    
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    batch_size = args.lm_batch_size
    # Show the training loss with every epoch
    logging_steps = len(train_dataset_seen) // batch_size
    print(f"Using model...{model_name}")
    print("Finetuning model...")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # print("Max train token", max(max(max(i) for i in batch['input_ids']) for batch in train_dataloader))
    # print("Max test token", max(max(max(i) for i in batch['input_ids']) for batch in test_dataloader))
    # print("Tokenizer vocab size", len(tokenizer))
    # print("Model vocab size", model.config.vocab_size)


    out_dir = f"{args.out_dir}"
    training_args = TrainingArguments(
            output_dir=out_dir,
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
            load_best_model_at_end=True
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_seen,
        eval_dataset=test_dataset_seen,
        tokenizer=tokenizer
    )
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, transformers.integrations.NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)

    # input("Press enter to continue...")
    eval_results = trainer.evaluate()
    print(f">>> Perplexity (before training): {math.exp(eval_results['eval_loss']):.2f}")
    results['train seen (before)'] = {'perplexity': math.exp(eval_results['eval_loss']), 'eval_loss': eval_results['eval_loss']}
    # results['train seen (before)'] = {'perplexity':0.1, 'eval_loss': 0.1}

    trainer.train()
    trainer.save_model()
    eval_results = trainer.evaluate()
    print("Evaluation loss: ", eval_results['eval_loss'])
    print(f">>> Perplexity (after training): {math.exp(eval_results['eval_loss']):.2f}")
    results['train seen (after)'] = {'perplexity': math.exp(eval_results['eval_loss']), 'eval_loss': eval_results['eval_loss']}
    # results['train seen (after)'] = {'perplexity':0.1, 'eval_loss': 0.1}
    model = trainer.model

    mask_filler = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0)
    accuracy = data_utils.get_accuracy(tokenizer, mask_filler, test_dataset_seen)
    print("Accuracy on masked tokens in training graphs: ", accuracy)
    results['train seen (after)'] = {'accuracy': accuracy}
    # results['train seen (after)']['accuracy'] = 0.1

    accuracy = data_utils.get_accuracy(tokenizer, mask_filler, train_dataset_unseen)
    print("Accuracy on masked tokens in unseen graphs train: ", accuracy)
    results['unseen train (before)'] = {'accuracy': accuracy}
    # results['unseen train (before)'] = {'accuracy': 0.1}

    accuracy = data_utils.get_accuracy(tokenizer, mask_filler, test_dataset_unseen)
    print("Accuracy on masked tokens in unseen graphs test: ", accuracy)
    results['unseen test (before)'] = {'accuracy': accuracy}
    # results['unseen test (before)'] = {'accuracy': 0.1}

    return results

# def main():
#     args = parse_args()
#     data_utils.set_seed(args.seed)
#     seen_graphs, unseen_graphs, _ = data_utils.get_graphs_data(args)
#     print(train_mlm(seen_graphs, unseen_graphs, args))

# # main()
