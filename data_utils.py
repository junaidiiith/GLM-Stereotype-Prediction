from transformers import AutoTokenizer, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import evaluate
from transformers import GPT2ForSequenceClassification, GPT2Config, AutoModelForSequenceClassification
import dgl
from sklearn.metrics import accuracy_score
import torch
import re
import random
from tqdm.auto import tqdm
import networkx as nx
from collections import Counter, defaultdict
import numpy as np
from graph_utils import get_node_text_triples

from utils import ontouml_meta_properties


vector_file_prefix = "word2vec_models/vectors_"


SEP = "[SEP]"
STEREOTYPE = "[STEREOTYPE]"

e_s = {'rel': 'relates', 'gen': 'generalizes'}
remove_extra_spaces = lambda txt: re.sub(r'\s+', ' ', txt.strip())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frequent_stereotypes = ['kind', 'subkind', 'phase', 'role', 'category', 'mixin', 'rolemixin', 'phasemixin']


class EncodingsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        output = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if 'token_type_ids' in self.encodings:
            output['token_type_ids'] = self.encodings['token_type_ids'][idx]
        if 'labels' in self.encodings:
            output['labels'] = self.encodings['labels'][idx]
        return output


accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def init_glove_model(model_type):
    f_vec = f"{vector_file_prefix}{model_type}.txt"
    vocab, W = init_glove(f_vec)
    return vocab, W


def get_encoding_size(triples, tokenizer, allowed_perc=0.005):
    single_encodings_map = defaultdict(int)
    if isinstance(triples[0], tuple):
        triples = [t[0] for t in triples]
        
    for triple in triples:
        single_encodings_map[len(tokenizer(triple)['input_ids'])] += 1

    values = sorted(single_encodings_map.items(), key=lambda x: x[0], reverse=True)
    for sz, _ in values:
        truncations = sum(v for k, v in single_encodings_map.items() if k >= sz)
        perc = truncations / len(triples)
        if perc >= allowed_perc:
            return sz, truncations, values


def get_accuracy(tokenizer, mask_filler_pipe, dataset):
    decoded_texts = list()
    for data in dataset:
        decoded_texts.append((data['labels'], tokenizer.decode(data['input_ids'])))
    count = 0
    all_preds, all_actuals = list(), list()
    for txt_labels, txt_batch in tqdm(decoded_texts):
        try:
            fill_out = mask_filler_pipe(txt_batch)
            preds = [x[0]['token'] for x in fill_out]
            actuals = txt_labels[torch.where(txt_labels != -100)].tolist()
            all_preds += preds
            all_actuals += actuals
            count += preds == actuals
        except:
            continue
  
    accuracy = count / len(decoded_texts)
    masked_accuracy = accuracy_score(all_actuals, all_preds)
    return f"{accuracy:.4f}", f"{masked_accuracy:.4f}"


def update_encodings(encodings, current_token, update_token, pad_token="[PAD]"):
    updated_encodings = {k: list() for k in ['input_ids', 'attention_mask', 'labels']}
    for ips in tqdm(encodings['input_ids'], desc='Adding [MASK] tokens'):
        input_ids, attention_mask, labels = get_updated_encoding(ips, current_token, update_token)
        updated_encodings['input_ids'].append(torch.tensor(input_ids))
        updated_encodings['attention_mask'].append(torch.tensor(attention_mask))
        updated_encodings['labels'].append(torch.tensor(labels))

    updated_encodings['input_ids'] = pad_1d_tensors(updated_encodings['input_ids'], pad_token)
    updated_encodings['attention_mask'] = pad_1d_tensors(updated_encodings['attention_mask'], 0)
    updated_encodings['labels'] = pad_1d_tensors(updated_encodings['labels'], -100)
    return updated_encodings


def get_updated_encoding(input_ids, current_token, update_token):
    i = 0
    updated_input_ids, labels = list(), list()
    while i < len(input_ids):
        curr_token = input_ids[i]
        if curr_token == current_token:
            i += 1
            while i < len(input_ids) and input_ids[i] != current_token:
                labels.append(input_ids[i])
                updated_input_ids.append(update_token)
                i += 1
        else:
            updated_input_ids.append(curr_token)
            labels.append(-100)
        i += 1
    attention_mask = [1] * len(updated_input_ids)
    return updated_input_ids, attention_mask, labels


def init_glove(vectors_file, normalize=False):
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    words = list(vectors.keys())
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    inv_vocab = {idx: w for idx, w in enumerate(words)}
    vector_dim = len(vectors[inv_vocab[0]])
    W_ = np.zeros((vocab_size, vector_dim))

    for word, v_ in vectors.items():
        if word == '<unk>':
            continue
        W_[vocab[word], :] = v_

    if normalize:
        d = (np.sum(W_ ** 2, 1) ** 0.5)
        W_ = (W_.T / d).T
    return vocab, W_


def get_w2v_embeddings(graph, w2v_model, uncased=True):
    vocab, W = w2v_model
    # embed_dim = W.shape[1]
    # node_embeddings = torch.randn(len(graph.nodes), embed_dim)
    # return node_embeddings
    node_embeddings = list()
    for n in graph.nodes:
        name = graph.nodes[n]['name'] if graph.nodes[n]['name'] != "Null" else "relates"
        name = name.lower()
        node_type = graph.nodes[n]['type'] if not uncased else graph.nodes[n]['type'].lower()
        name += f" {node_type}" 
        name_parts = name.split()
        embeddings = list()
        for part in name_parts:
            if part in vocab:
                embeddings.append(W[vocab[part]])
            else:
                embeddings.append(np.zeros(W.shape[1]))
        if len(embeddings) == 0:
            embeddings.append(np.zeros(W.shape[1]))
    
        embeddings = np.array(embeddings)
        embeddings = np.mean(embeddings, axis=0)

        meta_prop_vector = list()
        for prop in ontouml_meta_properties:
            key_value = str(graph.nodes[n][prop]).lower()
            if prop == 'type':
                value = 1 if key_value == 'class' else 2
            elif prop == 'aggregationKind':
                value = 1 if key_value == 'composite' else (2 if key_value == 'shared' else 0)
            elif prop in graph.nodes[n]:
                value = 1 if key_value == True else 0
            else:
                value = 0
            meta_prop_vector.append(value)
        
        meta_prop_vector = np.array(meta_prop_vector)
        embeddings = np.concatenate((embeddings, meta_prop_vector))
        node_embeddings.append(embeddings)
    node_embeddings = np.array(node_embeddings)
    return torch.from_numpy(node_embeddings)


def get_bert_embeddings(nxg, tokenizer_model, distance, only_name=False):
    tokenizer, model = tokenizer_model
    node_texts = get_node_text_triples(nxg, distance=distance, only_name=only_name)
    dataset = get_triples_dataset(node_texts, tokenizer, max_length=512)
    # node_embeddings = torch.randn(len(nxg.nodes), 768)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    node_embeddings = get_node_embeddings(model, dataloader)
    return node_embeddings


def get_w2v_embeddings_with_context(graph, w2v_model, distance, only_name=False):
    vocab, W = w2v_model
    node_texts = get_node_text_triples(graph, distance=distance, only_name=only_name)
    node_embeddings = list()
    for node_text in node_texts:
        name_parts = node_text.lower().split()
        embeddings = list()
        for part in name_parts:
            if part in vocab:
                embeddings.append(W[vocab[part]])
            else:
                embeddings.append(np.zeros(W.shape[1]))
        if len(embeddings) == 0:
            embeddings.append(np.zeros(W.shape[1]))
    
        embeddings = np.array(embeddings)
        embeddings = np.mean(embeddings, axis=0)
        node_embeddings.append(embeddings)

    node_embeddings = np.array(node_embeddings)
    return torch.from_numpy(node_embeddings)


def get_bert_tokenizer_model(model_name, num_classes):
    tokenizer = get_tokenizer(model_name)
    model = get_classification_model(model_name, num_classes, tokenizer)
    if model.device != device:
        model.to(device)
    return tokenizer, model



def create_dgl_from_nx(nxg, lbl_encoder, node_embeddings):
    dgl_graph = dgl.from_networkx(nxg)
    dgl_graph = dgl.add_self_loop(dgl_graph)
    dgl_graph.ndata['feat'] = node_embeddings

    labels = [lbl_encoder[nxg.nodes[node]['stereotype']] if 'masked' in nxg.nodes[node] else -1 for node in nxg.nodes]
    dgl_graph.ndata['labels'] = torch.tensor(labels)

    train_nodes = [node for node in nxg.nodes if 'masked' in nxg.nodes[node] and not nxg.nodes[node]['masked']]
    test_nodes = [node for node in nxg.nodes if 'masked' in nxg.nodes[node] and nxg.nodes[node]['masked']]

    train_mask = torch.zeros(len(nxg.nodes), dtype=torch.bool)
    train_mask[train_nodes] = True
    test_mask = torch.zeros(len(nxg.nodes), dtype=torch.bool)
    test_mask[test_nodes] = True

    dgl_graph.ndata['train_mask'] = train_mask
    dgl_graph.ndata['test_mask'] = test_mask
    return dgl_graph


def get_embedding_model(model_name, label_encoder):
    if 'bert' in model_name:
        print("Loading bert model...", model_name)
        model = get_bert_tokenizer_model(model_name, len(label_encoder))
        print("Bert model loaded.", model_name)
    else:
        print("Loading word2vec model...", model_name)
        model = init_glove_model(model_name)
        print("Word2vec model loaded.", model_name)
    return model


def create_dgl_graphs(graphs, lbl_encoder, model_name, distance, use_context=False):
    graphs = [nx.convert_node_labels_to_integers(g) for g, _ in graphs]
    graph_node_embeddings = get_graph_node_embeddings(graphs, lbl_encoder, model_name, distance, use_context=use_context)
    dgl_graphs = list()
    for graph, node_embeddings in zip(graphs, graph_node_embeddings):
        dgl_graph = create_dgl_from_nx(graph, lbl_encoder, node_embeddings)
        dgl_graphs.append(dgl_graph)
    
    return dgl_graphs


def get_graph_node_embeddings(graphs, lbl_encoder, model_name, distance, use_context=False):
    graph_node_embeddings = list()
    model = get_embedding_model(model_name, lbl_encoder)
    for graph in tqdm(graphs):
        node_embeddings = get_bert_embeddings(graph, model, distance) if 'bert' in model_name\
              else (get_w2v_embeddings(graph, model) if not use_context else get_w2v_embeddings_with_context(graph, model, distance))
        graph_node_embeddings.append(node_embeddings)
    return graph_node_embeddings


def set_seed(seed):
    print("Setting seed to", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_classification_model(model_name, num_labels, tokenizer):
    if 'bert' in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=num_labels)
        tokenizer.padding_side = "left"
        model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=model_config)
        model.resize_token_embeddings(len(tokenizer)) 
        model.config.pad_token_id = model.config.eos_token_id
        
    return model


def get_node_embeddings(model, dataloader):
    embedding_keys = ['input_ids', 'attention_mask']
    node_embeddings = list()
    with torch.no_grad():
        # for batch in tqdm(dataloader):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in embedding_keys}
            outputs = model.base_model(**batch)
            try:
                node_embeddings.append(outputs.pooler_output.detach().cpu())
            except AttributeError:
                node_embeddings.append(outputs.last_hidden_state[:, 0, :].detach().cpu())
    return torch.cat(node_embeddings, dim=0)


def get_label_encoder(graphs, exclude_limit):
    stereotypes = defaultdict(int)
    for g, _ in graphs:
        for node in g.nodes:
            if 'stereotype' in g.nodes[node]:
                stereotypes[g.nodes[node]['stereotype']] += 1


    if exclude_limit != -1:
        stereotypes_classes = [stereotype for stereotype, _ in filter(lambda x: x[1] > exclude_limit, stereotypes.items())]
    else:
        stereotypes_classes = [stereotype for stereotype, _ in filter(lambda x: x[0] in frequent_stereotypes, stereotypes.items())]
    # print(len(stereotypes_classes))
    label_encoder = {label: i for i, label in enumerate(stereotypes_classes)}
    return label_encoder


def get_tokenizer(model_name):
    if 'bert' in model_name:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    # tokenizer.add_special_tokens({'additional_special_tokens': [PATH_SEP]})
    
    return tokenizer


def get_graphs_info(graphs):
    masked = sum(1 for g, _ in graphs for n in g.nodes if 'masked' in g.nodes[n] and g.nodes[n]['masked'])
    not_masked = sum(1 for g, _ in graphs for n in g.nodes if 'masked' in g.nodes[n] and not g.nodes[n]['masked'])
    total_stereotypes = masked + not_masked
    total_nodes = sum(1 for g, _ in graphs for n in g.nodes)
    # print(f'Masked nodes: {masked}, Unmasked nodes: {not_masked}, Total stereotype nodes: {total_stereotypes}, Total nodes: {total_nodes}')
    # print(f'Percentage of masked nodes: {masked/total_stereotypes:.2f}')
    # print(f'Percentage of unmasked nodes: {not_masked/total_stereotypes:.2f}')
    info = {
        'num_nodes': total_nodes,
        'num_stereotype_nodes': total_stereotypes,
        'num_masked_nodes': masked,
        'num_unmasked_nodes': not_masked,
        'num_graphs': len(graphs),
    }
    return info


def pad_1d_tensors(tensors_list, padding_token):
    max_length = max(tensor.shape[0] for tensor in tensors_list)
    padded_tensors = []

    # Pad each tensor to match the maximum size along the specified dimension
    for tensor in tensors_list:
        sizes_to_pad = [max_length - s for s in tensor.shape]
        padding = torch.full(sizes_to_pad, padding_token, dtype=tensor.dtype)
        padded_tensor = torch.cat([tensor, padding], dim=0)
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors)


def get_triples_dataset(triples, tokenizer, max_length):
    max_length = max_length if max_length < 512 else 512
    encodings = tokenizer(triples, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    dataset = EncodingsDataset(encodings)
    return dataset


def get_predictions_distribution(preds, acts):
    correct_preds = [preds[i] for i in range(len(preds)) if preds[i] == acts[i] if acts[i] != -1]
    incorrect_preds = [preds[i] for i in range(len(preds)) if preds[i] != acts[i] if acts[i] != -1]
    correct_preds = Counter(correct_preds)
    incorrect_preds = Counter(incorrect_preds)
    predictions = {
        'correct': correct_preds,
        'incorrect': incorrect_preds,
    }
    return predictions
