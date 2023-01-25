import logging
import os
import pandas as pd
from pytorch_lightning import LightningModule, seed_everything
import torch.nn.functional as F
import torch
from torch.nn import Dropout, Linear, ReLU
from torch.optim import AdamW
import torch_geometric
from torch_geometric.nn import SAGEConv, GCNConv, GATv2Conv, Sequential
from torchmetrics import Accuracy, Precision, Recall, F1Score

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
seed_everything(42)

class LightningGNN(LightningModule):
    """
    Pytorch Lightning version of the GNN for parallelization over GPUs.
    """
    def __init__(self, 
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        train_batch_size: int = 256,
        eval_batch_size: int = 256,
        **kwargs):
        """
        Initializing the GNN model with hyperparameters, tokenizers and evaluation metrics.
        """
        super(LightningGNN, self).__init__()
        self.save_hyperparameters()
        self.eval_file = None
        self.gnn_type = kwargs["gnn_type"] \
                    if "gnn_type" in kwargs.keys() else "GAT"
        self.num_features = kwargs["num_features"] \
                    if "num_features" in kwargs.keys() else 3
        self.num_labels = kwargs["num_labels"] \
                    if "num_labels" in kwargs.keys() else 2

        # hidden layer node features
        self.hidden = 512 

        if self.gnn_type == "SAGE":
            gnn_model = SAGEConv
        elif self.gnn_type == "GCN":
            gnn_model = GCNConv
        elif self.gnn_type == "GAT":
            gnn_model = GATv2Conv
        else:
            raise ValueError("Please provide a gnn_type in ['SAGE', 'GCN', 'GAT'])")

        self.gnn_layer1 = gnn_model(self.num_features, self.hidden)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(p=0.5)
        
        self.gnn_layer2 = gnn_model(self.hidden, self.hidden)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(p=0.5)
        
        self.gnn_layer3 = gnn_model(self.hidden, self.hidden)
        self.relu3 = ReLU()
        self.dropout3 = Dropout(p=0.5)

        self.gnn_layer4 = gnn_model(self.hidden, self.hidden)
        self.relu4 = ReLU()
        self.dropout4 = Dropout(p=0.5)

        self.classifier = Linear(2*self.hidden, self.num_labels)
        self.metrics = {
                        "Accuracy": Accuracy(num_classes=self.num_labels), 
                        "Weighted-Precision": Precision(num_classes=self.num_labels,average="weighted"),
                        "Weighted-Recall": Recall(num_classes=self.num_labels,average="weighted"),
                        "Weighted-F1": F1Score(num_classes=self.num_labels, average="weighted"),
                        "Macro-F1": F1Score(num_classes=self.num_labels, average="macro"),
                        }

    def forward(self, x, edge_index, batch_index, edge_label_index):
        """
        Forward pass over the GNN model.
        Returns: logits (torch.Tensor) of a language model.
        """
        z_graph = self.gnn_layer1(x, edge_index)
        z_graph = self.relu1(z_graph)
        z_graph = self.dropout1(z_graph)

        z_graph = self.gnn_layer2(z_graph, edge_index)
        z_graph = self.relu2(z_graph)
        z_graph = self.dropout2(z_graph)

        z_graph = self.gnn_layer3(z_graph, edge_index)
        z_graph = self.relu3(z_graph)
        z_graph = self.dropout3(z_graph)

        z_graph = self.gnn_layer4(z_graph, edge_index)
        z_graph = self.relu4(z_graph)
        z_graph = self.dropout4(z_graph)

        srcs, dsts = edge_label_index
        z_out = self.classifier(torch.cat([z_graph[srcs],z_graph[dsts]],dim=-1))
        return z_out

    def training_step(self, batch, batch_index):
        """
        Training step computes loss over a single batch.
        Loss function is Binary Cross Entropy.
        Returns: loss (single tensor with backward method) over a single batch.
        """
        x, edge_index, edge_label_index = batch.x, batch.edge_index, batch.edge_label_index
        logits = self.forward(x, edge_index, batch_index, edge_label_index)
        loss_fct =  torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1,self.num_labels), batch.edge_label.float().view(-1,self.num_labels))
        return loss

    def validation_step(self, batch, batch_index):
        """
        Validation step computes loss and prediction labels over a single validation batch.
        Loss function is Binary Cross Entropy.
        Returns: dictionary with keys "loss", "preds" and "labels" and corresponding values
                of loss tensor, model prediction labels and ground truth labels, respectively.
        """
        x, edge_index, edge_label_index = batch.x, batch.edge_index, batch.edge_label_index
        logits = self.forward(x, edge_index, batch_index, edge_label_index)
        loss_fct =  torch.nn.BCEWithLogitsLoss()
        val_loss = loss_fct(logits.view(-1,self.num_labels), batch.edge_label.float().view(-1,self.num_labels))
        preds = torch.argmax(logits, axis=1)
        labels = torch.argmax(batch.edge_label, axis=1)
        return {"loss": val_loss, "preds": preds, "labels": labels}


    def validation_epoch_end(self, outputs):
        """
        Method to aggregate the loss, prediction labels and ground truth labels over all the validation steps
        to finally calculate the evaluation metrics and log it to the defined logger. 
        """
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        metrics_dict = {}
        for key,value in self.metrics.items():
            metrics_dict[key] = value(preds,labels)
        self.log_dict(metrics_dict, prog_bar=True)
        self.logger.log_metrics(metrics_dict)    

    def test_step(self, batch, batch_index):
        """
        Test step computes loss and prediction labels over a single test batch.
        Loss function is Binary Cross Entropy.
        Returns: dictionary with keys "edge_label_index", "loss", "preds", "labels", "logits" and corresponding 
                values of edge label index information, loss tensor, model prediction labels, ground truth labels, 
                and logit tensors, respectively.
        """
        x, edge_index, edge_label_index = batch.x, batch.edge_index, batch.edge_label_index
        logits = self.forward(x, edge_index, batch_index, edge_label_index)
        loss_fct =  torch.nn.BCEWithLogitsLoss()
        val_loss = loss_fct(logits.view(-1,self.num_labels), batch.edge_label.float().view(-1,self.num_labels))
        preds = torch.argmax(logits, axis=1)
        labels = torch.argmax(batch.edge_label, axis=1)
        return {"edge_label_index":edge_label_index, "loss": val_loss, "preds": preds, "labels": labels, "logits": logits}

    
    def test_epoch_end(self, outputs):
        """
        Method to aggregate the edge label indices, loss tensor, model prediction labels, ground truth labels, 
        and logit tensors over all the test steps to finally calculate the evaluation metrics and export to 
        the defined eval file. 
        """
        edge_label_index = torch.cat([x["edge_label_index"] for x in outputs],dim=1).detach().cpu()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        logits = torch.cat([x["logits"] for x in outputs],dim=0).detach().cpu()
        metrics_dict = {}
        for key,value in self.metrics.items():
            metrics_dict[key] = value(preds,labels)
        self.log_dict(metrics_dict, prog_bar=True)
        df = pd.DataFrame(logits.numpy(), columns = ['p_e','p_s','p_c','p_i'])
        df["node0"] = edge_label_index.numpy()[0]
        df["node1"] = edge_label_index.numpy()[1]
        df["labels"] = labels.numpy()
        df["preds"] = preds.numpy()
        if os.path.exists(f"{self.eval_file}.csv"):
            df.to_csv(f"{self.eval_file}.csv",mode='a',index=False, header=False)
        else:
            df.to_csv(f"{self.eval_file}.csv",index=False)

    def configure_optimizers(self):
        """Prepare optimizer with learning rate and adam epsilon"""
        return AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)