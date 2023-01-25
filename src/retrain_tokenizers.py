from cocolm.cocolm_tokenizer import COCOLMTokenizer
from lm_config import TRANSFORMERS_CONFIG, SPECIAL_TOKENS
import logging
import pandas as pd
from transformers import AutoTokenizer

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class TokenizerTrainer(object):
    """
    Special Tokenizer class to include special tokens to handle different product
    information; Special tokens are defined in lm_config.py
    """
    def _process_corpus(self,dataset):
        """
        Corpus processor to iterate over different columns of the product dataset.
        """
        dataset = dataset.fillna("")
        queries = dataset["keyword"].values
        products = dataset["title"].values
        descriptions = dataset["Description"].values
        bullet_points = dataset["BulletPoint"].values
        brands = dataset["Brand"].values
        colors = dataset["ColorName"].values
        parts = [queries,products,descriptions,bullet_points,brands,colors]
        sentences = [sentence for part in parts for sentence in part]
        for start_idx in range(0, len(sentences), 1000):
            samples = sentences[start_idx : start_idx + 1000]
            yield samples

    def locale_train(self,locale,dataset):
        """
        Locale-specific tokenizer training.
        """
        for model in TRANSFORMERS_CONFIG[locale]:
            logging.info(f"Training tokenizer {model} for locale {locale}")
            model_tag = model.split("/")[-1]
            if "cocolm" in model:
                new_tokenizer = COCOLMTokenizer(vocab_file="./cocolm_tokenize/sp.model",
                                                 dict_file="./cocolm_tokenize/dict.txt")
            else:
                old_tokenizer = AutoTokenizer.from_pretrained(model)
                training_corpus = self._process_corpus(dataset)
                new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
                new_tokenizer.add_tokens(list(SPECIAL_TOKENS.values()),special_tokens=True)
                new_tokenizer.save_pretrained(f"../saved_states/{locale}_{model_tag}.tokenizer")
    
    def locale_test(self,locale,model,test_samples):
        """
        Locale-specific tokenizer testing.
        """
        logging.info(f"Testing tokenizer {model} for locale {locale}")
        model_tag = model.split("/")[-1]
        if "cocolm" in model:
            new_tokenizer = COCOLMTokenizer(vocab_file="./cocolm_tokenize/sp.model", dict_file="./cocolm_tokenize/dict.txt")
        else:
            new_tokenizer = AutoTokenizer.from_pretrained(f"../saved_states/{locale}_{model_tag}.tokenizer")
        tokenized = []
        sentences = self._process_corpus(test_samples)
        for sentence in sentences:
            tokenized.extend([new_tokenizer.tokenize(sent) for sent in sentence])
        return tokenized
        
if __name__=="__main__":
    logger.info("Training and Testing Tokenizers on ESCI dataset.")
    locales = ["us","es","jp"]
    for locale in locales:
        dataset = pd.read_parquet(f"../esci_datasets/esci_text_{locale}.parquet")
        train_samples = dataset[dataset["split"]=="train"]
        test_samples = dataset[dataset["split"]=="test"].iloc[:10]
        tokenizer = TokenizerTrainer()
        tokenizer.locale_train(locale,train_samples)
        for model in TRANSFORMERS_CONFIG[locale]:
            test_output = tokenizer.locale_test(locale,model,test_samples)
            for tokenized in test_output: 
                print(tokenized)



        
