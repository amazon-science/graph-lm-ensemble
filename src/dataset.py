from clean import DeBertaCleanV2, ESClean, JSClean
from cocolm.cocolm_tokenizer import COCOLMTokenizer
from lm_config import ESCI_LABEL_MAP, SPECIAL_TOKENS, TRANSFORMERS_CONFIG
import logging
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class BaseDataset(Dataset):
    """
    BaseDataset for initializing model-specific tokenizers and 
    data cleaning functions for input sentences.
    """
    def __init__(
        self, df: pd.DataFrame, locale: str,
        model, clean, max_length) -> None:

        self.max_length = max_length
        self.df = df.fillna("")
        self.df.reset_index(inplace=True,drop=True)
        self.locale = locale
        self.model_tag = model.split("/")[-1]
        if "cocolm" in model:
            self.tokenizer = COCOLMTokenizer(vocab_file="./cocolm_tokenize/sp.model",
                                                 dict_file="./cocolm_tokenize/dict.txt")
        elif "bigbird" in model:
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model}")
        else:
            logging.info(f"Loading tokenizer from ../saved_states/{locale}_{self.model_tag}.tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(f"../saved_states/{locale}_{self.model_tag}.tokenizer")
        self.clean = clean()
        
    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def collate_fn(batch):
        ...

class ESCIDataset(BaseDataset):
    """
    Dataset to clean and construct dataset for LM Tokenizers.
    """
    def _get_special_token_ids(self):
        """
        Get Ids of added special tokens
        """
        return self.tokenizer.get_added_vocab()

    def __getitem__(self, index) -> Tuple:
        """
        Tokenize the input ESCI sample into the respective
        input_ids, attention_masks, special_token_pos and token_type_ids
        for input to huggingface language models.
        The query-product sample is processed as:
        [CLS]{query}[SEP]{product_title}[SEP]{product_id}[SEP]{product_brand}
        [SEP]{product_color_name}[SEP]{product_bullet_point}[SEP]{product_description}[SEP]

        Input:
        index: int; row number of the sample in pandas dataframe.
        Returns:
        feature: dict, for input to LM models
                        {"input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "special_token_pos": input_ids_pos,
                        "label": label}
        meta: dict, for tracking the sample predictions
                    {"product_id": ,
                     "sample_id":}
        """
        row = self.df.loc[index]
        query = self.tokenizer.encode(self.clean(row["keyword"]))
        
        product_title = self.tokenizer.encode(self.clean(row["title"]))
        product_id = self.tokenizer.encode(self.clean(row["DocumentId"]))
        product_description = self.tokenizer.encode(self.clean(row["Description"]))
        product_brand = self.tokenizer.encode(self.clean(row["Brand"]))
        product_color_name = self.tokenizer.encode(self.clean(row["ColorName"]))
        product_bullet_point = self.tokenizer.encode(self.clean(row["BulletPoint"]))   

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        special_token_id = self._get_special_token_ids()
    
        input_ids_pos = [1]
        special_token_flag = [special_token_id[SPECIAL_TOKENS["query"]]] if not(("cocolm" in self.model_tag) or ("bigbird" in self.model_tag)) else []
        input_ids = [cls_token_id] + special_token_flag + query[1:-1] + [sep_token_id]
        input_ids_pos.append(len(input_ids))

        special_token_flag = [special_token_id[SPECIAL_TOKENS["title"]]] if not(("cocolm" in self.model_tag) or ("bigbird" in self.model_tag)) else []
        input_ids += special_token_flag + product_title[1:-1] + [sep_token_id]
        input_ids_pos.append(len(input_ids))

        special_token_flag = [special_token_id[SPECIAL_TOKENS["id"]]] if not(("cocolm" in self.model_tag) or ("bigbird" in self.model_tag)) else []
        input_ids += special_token_flag + product_id[1:-1] + [sep_token_id]
        input_ids_pos.append(len(input_ids))

        special_token_flag = [special_token_id[SPECIAL_TOKENS["brand"]]] if not(("cocolm" in self.model_tag) or ("bigbird" in self.model_tag)) else []
        input_ids += special_token_flag + product_brand[1:-1] + [sep_token_id]
        input_ids_pos.append(len(input_ids))

        special_token_flag = [special_token_id[SPECIAL_TOKENS["color"]]] if not(("cocolm" in self.model_tag) or ("bigbird" in self.model_tag)) else []
        input_ids += special_token_flag + product_color_name[1:-1] + [sep_token_id]
        input_ids_pos.append(len(input_ids))

        special_token_flag = [special_token_id[SPECIAL_TOKENS["bullet"]]] if not(("cocolm" in self.model_tag) or ("bigbird" in self.model_tag)) else []
        input_ids += special_token_flag + product_bullet_point[1:-1] + [sep_token_id]
        input_ids_pos.append(len(input_ids))

        special_token_flag = [special_token_id[SPECIAL_TOKENS["description"]]] if not(("cocolm" in self.model_tag) or ("bigbird" in self.model_tag)) else []
        input_ids += special_token_flag + product_description[1:-1]
        input_ids = input_ids[:self.max_length-1] + [sep_token_id]
        for i in range(len(input_ids_pos)):
            if input_ids_pos[i] >= self.max_length:
                if input_ids_pos[-2] < self.max_length:
                    input_ids_pos[i] = input_ids_pos[-2]
                elif input_ids_pos[1] < self.max_length:
                    input_ids_pos[i] = input_ids_pos[1]
                else:
                    input_ids_pos[i] = self.max_length - 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids_pos = torch.tensor(input_ids_pos, dtype=torch.long)[None]
        attention_mask = torch.ones_like(input_ids)
        label = torch.tensor(ESCI_LABEL_MAP[row['label']], dtype=torch.long)
        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_token_pos": input_ids_pos,
            "label": label
        }

        if 'bigbird' in self.model_tag:
            feature['token_type_ids'] = torch.zeros_like(input_ids)

        meta = {
            "product_id": row['DocumentId'],
            "sample_id": row['sample_id'],
        }
        return feature, meta

    @staticmethod
    def collate_fn(batch: List) -> dict:
        """
        Collate the individual items of the batch together
        into a single batch object for processing.
        """
        features = {}
        features["input_ids"] = pad_sequence(
            [x[0]["input_ids"] for x in batch],
            batch_first=True,
            padding_value=0,
        )
        features["attention_mask"] = pad_sequence(
            [x[0]["attention_mask"] for x in batch],
            batch_first=True,
        )
        features["special_token_pos"] = torch.cat(
            [x[0]["special_token_pos"] for x in batch]
        )
        features["label"] = torch.cat(
            [x[0]["label"].unsqueeze(dim=0) for x in batch]
        )
        if 'token_type_ids' in batch[0][0]:
            features["token_type_ids"] = pad_sequence(
                [x[0]["token_type_ids"] for x in batch],
                batch_first=True,
            )

        meta = {}
        meta["product_id"] = [x[1]["product_id"] for x in batch]
        meta["sample_id"] = [x[1]["sample_id"] for x in batch]
        return {"features": features, "meta": meta}

if __name__ == "__main__":
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
            dataset = ESCIDataset(df,locale,model,cleaners[locale],max_length=512)
            dataloader = DataLoader(dataset, batch_size=64, num_workers=8, 
                                    drop_last=False, pin_memory=True,
                                    collate_fn=dataset.collate_fn)
            logging.info(f"Locale: {locale}, Model: {model}")
            for batch in dataloader:
                print(batch)
                break