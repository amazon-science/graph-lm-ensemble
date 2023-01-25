from clean import DeBertaCleanV2, ESClean, JSClean
from lightning_dataset import  ESCIDataModule
from lm_config import TRANSFORMERS_CONFIG, DATASET_FILES, NUM_LABELS, MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from models import LMModel
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

if __name__=="__main__":
    cleaners = {
            "us": DeBertaCleanV2,
            "es": ESClean,
            "jp": JSClean,
        }
    for locale in DATASET_FILES:
        for model in TRANSFORMERS_CONFIG[locale]:
            df = pd.read_parquet(DATASET_FILES[locale])
            dm = ESCIDataModule(df,locale,model,cleaners[locale], 
                                MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE)
            dm.setup("fit")
            model_name = model.split("/")[-1]
            task_name = f"ESCI Classification Locale: {locale}, Model: {model_name}"
            wandb_logger_name = f"esci_{locale}_{model_name}"
            checkpoint_callback = ModelCheckpoint(
                                    dirpath=f'../model_checkpoints/esci_{locale}_{model_name}/',
                                    filename='ESCI-{epoch:02d}-{val_loss:.2f}')
            model = LMModel(
                model_name=model,
                num_labels=NUM_LABELS,
                task_name=task_name,
            )
            wandb_logger = WandbLogger(project=wandb_logger_name,log_model="all")
            trainer = Trainer(
                max_epochs=6,
                accelerator="gpu",
                devices=8 if torch.cuda.is_available() else None,
                callbacks=[checkpoint_callback],
                logger=wandb_logger
            )
            trainer.fit(model, datamodule=dm)
            trainer.validate(model, datamodule=dm)
            wandb_logger.finalize("success")
            wandb.finish()
