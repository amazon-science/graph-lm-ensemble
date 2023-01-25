from clean import DeBertaCleanV2, ESClean, JSClean
from dataset import ESCIDataset
from lm_config import TRANSFORMERS_CONFIG
import logging
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
seed_everything(42)

class ESCIDataModule(LightningDataModule):
    """
    Load the ESCI dataset into the Lightning Data Module for parallelization.
    """
    def __init__(
        self,
        df: pd.DataFrame, 
        locale: str,
        model: str, 
        clean,
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        """
        Initialize parameters for the dataloader.
        """

        super().__init__()
        self.df = df
        self.datasets = {}
        self.locale = locale
        self.model = model
        self.clean = clean
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def setup(self, stage: str):
        """
        Setup different splits of the dataset.
        """
        self.df = self.df.fillna("")
        train_val_dataset = self.df[self.df["split"]=="train"]
        np.random.seed(42)
        msk = np.random.rand(len(train_val_dataset)) < 0.8
        train_dataset = train_val_dataset[msk]
        val_dataset = train_val_dataset[~msk]
        test_dataset = self.df[self.df["split"]=="test"]
        self.datasets = {
            "train": ESCIDataset(train_dataset.reset_index(drop=True), self.locale, 
                                self.model, self.clean, self.max_seq_length),
            "validation": ESCIDataset(val_dataset.reset_index(drop=True), self.locale, 
                                self.model, self.clean, self.max_seq_length),
            "test": ESCIDataset(test_dataset.reset_index(drop=True), self.locale, 
                                self.model, self.clean, self.max_seq_length)
        }

    def prepare_data(self):
        pass

    def train_dataloader(self):
        """
        Construct the iterable train dataloader.
        """
        return DataLoader(self.datasets["train"], batch_size=self.train_batch_size, 
                        num_workers=8, drop_last=False, pin_memory=True,
                        collate_fn=self.datasets["train"].collate_fn)

    def val_dataloader(self):
        """
        Construct the iterable val dataloader.
        """
        return DataLoader(self.datasets["validation"], batch_size=self.eval_batch_size,
                        num_workers=8, drop_last=False, pin_memory=True,
                        collate_fn=self.datasets["validation"].collate_fn)

    def test_dataloader(self):
        """
        Construct the iterable test dataloader.
        """
        return DataLoader(self.datasets["test"], batch_size=self.eval_batch_size,
                        num_workers=8, drop_last=False, pin_memory=True,
                        collate_fn=self.datasets["test"].collate_fn)

if __name__=="__main__":
    dataset_files = {
        "us": "../esci_datasets/esci_text_us.parquet",
        "es": "../esci_datasets/esci_text_es.parquet",
        "jp": "../esci_datasets/esci_text_jp.parquet"
    }
    cleaners = {
        "us": DeBertaCleanV2,
        "es": ESClean,
        "jp": JSClean,
    }
    logging.info(f"Building a dataset from {dataset_files}")
    for locale in dataset_files:
        for model in TRANSFORMERS_CONFIG[locale]:
            df = pd.read_parquet(dataset_files[locale])
            dm = ESCIDataModule(df,locale,model,cleaners[locale], 512, 32, 32)
            dm.setup("fit")
            print(next(iter(dm.train_dataloader())))
            print(next(iter(dm.val_dataloader())))
            print(next(iter(dm.test_dataloader())))

