from graph_dataset import ESCIGraphDataset
import logging
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule, seed_everything
from torch_geometric.loader.link_neighbor_loader import LinkNeighborLoader

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
seed_everything(42)

class ESCIGraphDataModule(LightningDataModule):
    """
    Load the ESCI Graph dataset into the Lightning Data Module for parallelization.
    """
    def __init__(
        self,
        dataset,
        num_neighbors: int = 30,
        num_hops: int = 2,
        train_batch_size: int = 256,
        eval_batch_size: int = 256,
        **kwargs,
    ):
        """
        Initialize parameters for the dataloader.
        """
        super().__init__()
        self.dataset = dataset
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size


    def setup(self, stage: str):
        """
        Setup different splits of the dataset.
        """
        self.train_edge_index = self.dataset.edge_label_set["input_train_edges"]
        self.train_edge_label = self.dataset.edge_label_set["input_train_labels"]
        self.val_edge_index = self.dataset.edge_label_set["input_val_edges"]
        self.val_edge_label = self.dataset.edge_label_set["input_val_labels"]
        self.test_edge_index = self.dataset.edge_label_set["input_test_edges"]
        self.test_edge_label = self.dataset.edge_label_set["input_test_labels"]

    def prepare_data(self):
        pass

    def train_dataloader(self):
        """
        Construct the iterable train dataloader.
        """
        return LinkNeighborLoader(self.dataset.data, num_neighbors=[self.num_neighbors]*self.num_hops, 
                                  batch_size=self.train_batch_size, 
                                  edge_label_index=self.train_edge_index,
                                  edge_label=self.train_edge_label, num_workers=8)

    def val_dataloader(self):
        """
        Construct the iterable val dataloader.
        """
        return LinkNeighborLoader(self.dataset.data, num_neighbors=[self.num_neighbors]*self.num_hops, 
                                  batch_size=self.eval_batch_size, 
                                  edge_label_index=self.val_edge_index,
                                  edge_label=self.val_edge_label, num_workers=8)

    def test_dataloader(self):
        """
        Construct the iterable test dataloader.
        """
        return LinkNeighborLoader(self.dataset.data, num_neighbors=[self.num_neighbors]*self.num_hops, 
                                  batch_size=self.eval_batch_size, 
                                  edge_label_index=self.test_edge_index,
                                  edge_label=self.test_edge_label, num_workers=8)