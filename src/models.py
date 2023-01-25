from cocolm.configuration_cocolm import COCOLMConfig
from cocolm.modeling_cocolm import COCOLMForSequenceClassification
from lm_config import TRANSFORMERS_CONFIG
import logging
import os
import pandas as pd
from pytorch_lightning import LightningModule, seed_everything
import torch
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import AutoConfig, AutoModelForSequenceClassification,get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
seed_everything(42)

class LMModel(LightningModule):
    """
    Pytorch Lightning version of the language model for parallelization over GPUs.
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        task_name: str,
        eval_file: str = None,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        **kwargs,
    ):
        """
        Initializing the pertrained language model with hyperparameters, tokenizers and evaluation metrics.
        """
        super().__init__()

        self.save_hyperparameters()

        self.model_name = model_name
        self.task_name = task_name
        self.num_labels = num_labels
        self.eval_file = eval_file
        logging.info(f"Using {self.model_name} model with {num_labels} labels")
        if "cocolm" in self.model_name:
            self.config = COCOLMConfig.from_pretrained(self.model_name, num_labels=num_labels)
            self.model = COCOLMForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        self.metrics = {
                        "Accuracy": Accuracy(num_classes=self.num_labels), 
                        "Weighted-Precision": Precision(num_classes=self.num_labels,average="weighted"),
                        "Weighted-Recall": Recall(num_classes=self.num_labels,average="weighted"),
                        "Weighed-F1": F1Score(num_classes=self.num_labels, average="weighted"),
                        "Macro-F1": F1Score(num_classes=self.num_labels, average="macro")
                        }

    def forward(self, input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_token_pos: torch.Tensor):  
        """
        Forward pass over the language model.
        Returns: logits (torch.Tensor) of a language model.
        """
        return self.model(input_ids=input_ids, 
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        """
        Training step computes loss over a single batch.
        Loss function is Binary Cross Entropy.
        Returns: loss (single tensor with backward method) over a single batch.
        """
        features = batch["features"]
        meta = batch["meta"]
        input_ids = features["input_ids"]
        if "bigbird" in self.model_name:
            token_type_ids = features["token_type_ids"]
        else:
            token_type_ids = None 
        attention_mask = features["attention_mask"]
        special_token_pos = features["special_token_pos"]
        labels = features["label"]
        outputs = self(input_ids, token_type_ids,
                       attention_mask, special_token_pos)
        try:
            logits = outputs.logits
        except:
            logits = outputs[0]
        loss_fct =  torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.num_labels),labels.float().view(-1, self.num_labels))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step computes loss and prediction labels over a single validation batch.
        Loss function is Binary Cross Entropy.
        Returns: dictionary with keys "loss", "preds" and "labels" and corresponding values
                of loss tensor, model prediction labels and ground truth labels, respectively.
        """
        features = batch["features"]
        meta = batch["meta"]
        input_ids = features["input_ids"]
        if "bigbird" in self.model_name:
            token_type_ids = features["token_type_ids"]
        else:
            token_type_ids = None 
        attention_mask = features["attention_mask"]
        special_token_pos = features["special_token_pos"]
        labels = features["label"]
        outputs = self(input_ids, token_type_ids,
                       attention_mask, special_token_pos)
        try:
            logits = outputs.logits
        except:
            logits = outputs[0]
        loss_fct =  torch.nn.BCEWithLogitsLoss()
        val_loss = loss_fct(logits.view(-1, self.num_labels),labels.float().view(-1, self.num_labels))
        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = torch.argmax(features["label"], axis=1)

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

    def test_step(self, batch, batch_idx):
        """
        Test step computes loss and prediction labels over a single test batch.
        Loss function is Binary Cross Entropy.
        Returns: dictionary with keys "meta", "loss", "preds", "labels", "logits" and corresponding 
                values of meta information, loss tensor, model prediction labels, ground truth labels, 
                and logit tensors, respectively.
        """
        features = batch["features"]
        meta = batch["meta"]
        input_ids = features["input_ids"]
        if "bigbird" in self.model_name:
            token_type_ids = features["token_type_ids"]
        else:
            token_type_ids = None 
        attention_mask = features["attention_mask"]
        special_token_pos = features["special_token_pos"]
        labels = features["label"]
        outputs = self(input_ids, token_type_ids,
                       attention_mask, special_token_pos)
        try:
            logits = outputs.logits
        except:
            logits = outputs[0]
        loss_fct =  torch.nn.BCEWithLogitsLoss()
        val_loss = loss_fct(logits.view(-1, self.num_labels),labels.float().view(-1, self.num_labels))
        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = torch.argmax(features["label"], axis=1)

        return {"meta": meta, "loss": val_loss, "preds": preds, "labels": labels, "logits": logits}
    
    def test_epoch_end(self, outputs):
        """
        Method to aggregate the meta information, loss tensor, model prediction labels, ground truth labels, 
        and logit tensors over all the test steps to finally calculate the evaluation metrics and export to 
        the defined eval file. 
        """
        meta = torch.cat([torch.tensor(x["meta"]["sample_id"]) for x in outputs]).detach().cpu()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        logits = torch.cat([x["logits"] for x in outputs],dim=0).detach().cpu()
        metrics_dict = {}
        for key,value in self.metrics.items():
            metrics_dict[key] = value(preds,labels)
        self.log_dict(metrics_dict, prog_bar=True)
        df = pd.DataFrame(logits.numpy(), columns = ['p_e','p_s','p_c','p_i'])
        df["sample_id"] = meta.numpy()
        df["labels"] = labels.numpy()
        df["preds"] = preds.numpy()
        if os.path.exists(f"{self.eval_file}.csv"):
            df.to_csv(f"{self.eval_file}.csv",mode='a',index=False, header=False)
        else:
            df.to_csv(f"{self.eval_file}.csv",index=False)
        
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

if __name__=="__main__":
    task_name = "ESCI Classification"
    for locale in TRANSFORMERS_CONFIG:
        for model_name in TRANSFORMERS_CONFIG[locale]:
            init_model = LMModel(model_name,task_name,num_labels=4)