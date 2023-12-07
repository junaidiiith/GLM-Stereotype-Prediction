import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class GraphTrainer:
    def __init__(self, model, args) -> None:
        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.gnn_lr)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=10, num_training_steps=args.num_training_steps)
        self.criteria = nn.CrossEntropyLoss()
        self.edge2index = lambda g: torch.stack(list(g.edges())).contiguous()
        self.args = args
        print("GenericGraphTrainer initialized.")

    def run_epoch(self, dataloader, training=True):
        torch.set_grad_enabled(training)
        if training:
            self.model.train()
        else:
            self.model.eval()

        train_preds, train_acts = list(), list()
        test_preds, test_acts = list(), list()
        train_avg_loss, test_avg_loss = 0, 0
        _, topk_test_preds = list(), list()

        # for batch in tqdm(dataloader, total=len(dataloader), desc=f"{train_str} GNN"):
        for batch in dataloader:
            if training:
                self.optimizer.zero_grad()
                self.model.zero_grad()
            # input_ids, labels = batch
            if isinstance(batch, tuple):
                batch = batch[0]

            labels = batch.ndata['labels'].to(device)
            train_mask = batch.ndata['train_mask']
            test_mask = batch.ndata['test_mask']

            # Forward
            # logits = model(batched_graph, features)
            logits = self.get_logits(batch)

            # Compute prediction
            pred = logits.argmax(1)
            if logits[train_mask].shape[0] == 0:
                continue

            # Compute loss for training set nodes
            loss = self.criteria(logits[train_mask], labels[train_mask])
            train_avg_loss += loss.item()
            train_preds += pred[train_mask].tolist()
            train_acts += labels[train_mask].tolist()

            topk_test_preds += logits[test_mask].topk(min(8, logits.shape[-1])).indices.tolist()

            # Backward
            if training:
                loss.backward(torch.ones_like(loss))
                self.optimizer.step()
                # self.scheduler.step()

            if logits[test_mask].shape[0] == 0:
                continue
            # Compute loss for test set nodes
            test_loss = self.criteria(logits[test_mask], labels[test_mask])
            test_avg_loss += test_loss.item()
            test_preds += pred[test_mask].tolist()
            test_acts += labels[test_mask].tolist()

        
        train_acc = (np.array(train_preds) == np.array(train_acts)).mean()
        test_acc = (np.array(test_preds) == np.array(test_acts)).mean()

        # train_predictions_distribution = data_utils.get_predictions_distribution(train_preds, train_acts)
        # test_predictions_distribution = data_utils.get_predictions_distribution(test_preds, test_acts)

        train_avg_loss = np.log2(1 + train_avg_loss)
        test_avg_loss = np.log2(1 + test_avg_loss)
        torch.set_grad_enabled(True)
        output = {
            'train_acc': train_acc, 
            'train_loss': train_avg_loss,
            'test_acc': test_acc,
            'test_loss': test_avg_loss,
            # 'train_predictions_distribution': train_predictions_distribution,
            # 'test_predictions_distribution': test_predictions_distribution
            'top2acc': np.mean([1 if label in topk[:2] else 0 for topk, label in zip(topk_test_preds, test_acts)]),
            'top3acc': np.mean([1 if label in topk[:3] else 0 for topk, label in zip(topk_test_preds, test_acts)]),
            'top4acc': np.mean([1 if label in topk[:4] else 0 for topk, label in zip(topk_test_preds, test_acts)]),
            'top5acc': np.mean([1 if label in topk[:5] else 0 for topk, label in zip(topk_test_preds, test_acts)]),
            'top6acc': np.mean([1 if label in topk[:6] else 0 for topk, label in zip(topk_test_preds, test_acts)]),
        }

        return output

    def get_logits(self, batched_graph):
        self.model.zero_grad()
        x = batched_graph.ndata['feat'].to(device)
        model_name = type(self.model).__name__
        if "GNN" in model_name:
            edge_index = self.edge2index(batched_graph).to(device)
            x = x.float()
            logits = self.model(x, edge_index)
        else:
            logits = self.model(x)
        return logits


    def run_epochs(self, dataloader, num_epochs):
        max_val_acc, max_train_acc = 0, 0
        outputs = list()
        for _ in tqdm(range(num_epochs), desc="Epochs"):
        # for epoch in range(num_epochs):
            output = self.run_epoch(dataloader, training=True)
            
            train_acc, test_acc = output['train_acc'], output['test_acc']
            max_val_acc = max(max_val_acc, test_acc)
            max_train_acc = max(max_train_acc, train_acc)

            # if epoch % 50 == 0 and epoch > 0:
            # print(f"Epoch {epoch}: Train loss {output['train_loss']} and Test Loss {output['test_loss']}")
            # print(f"Train Accuracy: {train_acc} Test Accuracy: {test_acc}")
            outputs.append(output)
        
        print(f"Max Test Accuracy: {max_val_acc}")
        print(f"Max Train Accuracy: {max_train_acc}")
        max_output = max(outputs, key=lambda x: x['test_acc'])
        return max_output
        

    def validate(self, dataloader):
        output = self.run_epoch(dataloader, training=False)
        train_acc, test_acc = output['train_acc'], output['test_acc']
        print(f"Train loss {output['train_loss']} and Test Loss {output['test_loss']}")
        print(f"Train Accuracy: {train_acc} Test Accuracy: {test_acc}")
        return output
