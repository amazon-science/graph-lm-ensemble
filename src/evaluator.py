from clean import DeBertaCleanV2, ESClean, JSClean
from lightning_dataset import  ESCIDataModule
from lm_config import TRANSFORMERS_CONFIG, DATASET_FILES, NUM_LABELS, MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from models import LMModel
import os
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

if __name__=="__main__":
    cleaners = {
            "us": DeBertaCleanV2,
            "es": ESClean,
            "jp": JSClean,
        }
    for locale in DATASET_FILES:
        for model_n in TRANSFORMERS_CONFIG[locale]:
            for split in ["val","train","test"]:
                df = pd.read_parquet(DATASET_FILES[locale])
                dm = ESCIDataModule(df,locale,model_n,cleaners[locale], 
                                    MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE)
                dm.setup("evaluate")
                model_name = model_n.split("/")[-1]
                task_name = f"ESCI Classification Locale: {locale}, Model: {model_name}"
                eval_file_name = f"esci_{locale}_{model_name}"
                path = f"../model_checkpoints/{eval_file_name}"
                checkpoint = list(os.path.join(path,file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)))[0]
                model = LMModel(
                    model_name=model_n,
                    num_labels=NUM_LABELS,
                    task_name=task_name,
                )
                model = model.load_from_checkpoint(checkpoint)
                model = model.eval()
                evaluator = Trainer(
                    max_epochs=1,
                    accelerator="gpu",
                    devices=8
                )
                if not(os.path.exists(f"../evaluation_results/{eval_file_name}/")):
                    os.makedirs(f"../evaluation_results/{eval_file_name}/")
                model.eval_file = f"../evaluation_results/{eval_file_name}/{split}_preds"
                if os.path.exists(model.eval_file+".csv"):
                    continue
                if split == "train":
                    evaluator.test(model, dm.train_dataloader())
                elif split == "val":
                    evaluator.test(model, dm.val_dataloader())
                else:
                    evaluator.test(model, dm.test_dataloader())
            
