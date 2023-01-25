import gc
from graph_config import DATASET_FILES, GNN_MODELS, NUM_FEATURES, NUM_LABELS, RELATIONS
from graph_dataset import ESCIGraphDataset
from graph_lightning_dataset import ESCIGraphDataModule
from graph_models import LightningGNN
import logging
import os
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__=="__main__":
    for locale in DATASET_FILES:
        for relation in RELATIONS:
            for gnn_model in GNN_MODELS:
                for split in ["val","train","test"]:
                    logging.info(f"Running esci_{locale}_{relation}_{gnn_model}")
                    dataset = ESCIGraphDataset(root=dataset_files[locale],relation=relation)
                    dm = ESCIGraphDataModule(dataset)
                    dm.setup("evaluate")
                    task_name = f"ESCI Classification Locale: {locale}, Relation: {relation}, Model: {gnn_model}"
                    eval_file_name = f"esci_{locale}_{relation}_{gnn_model}"
                    path = f"../model_checkpoints/{eval_file_name}"
                    checkpoint = list(os.path.join(path,file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)))[0]
                    model = LightningGNN(
                        gnn_type=gnn_model,
                        num_features=NUM_FEATURES,
                        num_labels=NUM_LABELS
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
