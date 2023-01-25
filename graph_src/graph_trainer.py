import gc
from graph_models import LightningGNN
from graph_lightning_dataset import ESCIGraphDataModule
from graph_config import DATASET_FILES, GNN_MODELS, RELATIONS, NUM_FEATURES, NUM_LABELS
from graph_dataset import ESCIGraphDataset
import logging
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__=="__main__":
    for locale in DATASET_FILES:
        for relation in RELATIONS:
            for gnn_model in GNN_MODELS:
                logging.info(f"Running esci_{locale}_{relation}_{gnn_model}")
                dataset = ESCIGraphDataset(root=dataset_files[locale],relation=relation)
                logging.info(f"Done with making regular dataset")
                dm = ESCIGraphDataModule(dataset)
                logging.info(f"Done with making lightning dataset")
                dm.setup("fit")
                task_name = f"ESCI Classification Locale: {locale}, Relation: {relation}, Model: {gnn_model}"
                wandb_logger_name = f"esci_{locale}_{relation}_{gnn_model}"
                checkpoint_callback = ModelCheckpoint(
                                        dirpath=f'../model_checkpoints/esci_{locale}_{relation}_{gnn_model}/',
                                        filename='ESCI-GRAPH-{epoch:02d}-{val_loss:.2f}')
                model = LightningGNN(
                    gnn_type=gnn_model,
                    num_features=NUM_FEATURES,
                    num_labels=NUM_LABELS
                )
                wandb_logger = WandbLogger(project=wandb_logger_name,offline=False,log_model="all")
                trainer = Trainer(
                    max_epochs=20,
                    accelerator="auto",
                    devices=8 if torch.cuda.is_available() else None,
                    callbacks=[checkpoint_callback],
                    logger=wandb_logger
                )
                trainer.fit(model, datamodule=dm)
                trainer.validate(model, datamodule=dm)
                wandb_logger.finalize("success")
                wandb.finish()
