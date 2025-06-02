# Explainable Search Relevance using Graph-Enhanced Plug and Play Language Models

E-commerce search relevance is a challenging task as it involves understanding it with the appropriate products in the catalog. The problem has traditionally been addressed using language models (LMs) and graph neural networks (GNNs) to capture semantic and inter-product behavior signals, respectively. However, the rapid development of   new architectures has created a gap between research and practical adoption of these techniques. Evaluating the generalizability of these models for deployment requires extensive experimentation on complex, real-world datasets, which can be non-trivial and expensive. Furthermore, such models often operate on a latent space representations that are incomprehensible to humans, making it difficult to evaluate and compare the effectiveness of different models. This lack of interpretability hinders the development and adoption of new techniques in the field. To bridge this gap, we propose Plug and Play Graph LAnguage Model (PP-GLAM), an explainable ensemble of plug and play models. Our approach uses a modular framework with uniform data processing pipelines. It employs additive explanation metrics to independently decide whether to include (i) language model candidates, (ii) GNN model candidates,  
and (iii) inter-product relation types. For the task of search relevance, we show that PP-GLAM outperforms several state-of-the-art baselines as well as a proprietary model on real-world multilingual, multi-regional e-commerce datasets. To promote better model  
comprehensibility and adoption, we also provide an analysis of the explainability and computational complexity of our model. We also provide the public codebase, release a dataset of behavioral signals and provide deployment strategy for practical implementation.

![Overview of Plug and Play Graph LAnguage Model (PP-GLAM)](figs/ensemble_model.png)


## Environment Installation

```bash
# Please make sure torch-geometric, transformers
# and pytorch-lightning are compatible with each other
# and supported by your setup.
pip install pyg-lib torch-scatter torch-sparse
-f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html
pip install torch-geometric
pip install -r graph_src/requirements
pip install -r src/requirements
```

## Run
### Training PP-GLAM model
```bash
# Training language models.
cd src/
# Update parameters in lm_config.py
python retrain_tokenizers.py
python trainer.py

# Training graph models.
cd graph_src/
# Update parameters in graph_config.py
python graph_trainer.py

# Model Checkpoints are by-default stored in ../model_checkpoints/
# Use the models to get predictions over training dataset.
# Aggregate the features in a table (X_train,y_train).
```
```python
# Run ensemble model
import lightgbm
lgb_model = lgb.LGBMClassifier(
    objective = 'multiclass',
    class_weight = "balanced",
    boosting_type = "gbdt",
    num_iterations = 1500,
    learning_rate = 0.005,
    num_leaves = 25,
    max_depth = 25,
    min_data_in_leaf = 25,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 200,
    n_jobs = [#CPU_THREADS]
)
lgb_model.fit(X_train,y_train)
from joblib import dump
dump(lgb_model, 'pp_glam_ensemble_model.joblib')
```

### Evaluating PP-GLAM model
```bash
# Evaluating language models.
cd src/
# Update parameters in lm_config.py
python evaluator.py

# Evaluating graph models.
cd graph_src/
# Update parameters in graph_config.py
python graph_evaluator.py

# Model Checkpoints are by-default stored in ../model_checkpoints/
# Use the models to get predictions over training dataset.
# Aggregate them in a table X_test.
```
```python
# Run ensemble model
from joblib import load
lgb_model = load('pp_glam_ensemble_model.joblib')
y_test = lgb_model.predict(X_test)
```
## Dataset Processing
The paper uses dataset provided here; https://github.com/amazon-science/esci-data

Language Model:
```bash
cd src
python dataset.py
```
Graphs:
```bash
cd graph_src
python graph_dataset.py
```
Preprocessed datasets coming soon.

## Paper
```
@inproceedings{10.1145/3589335.3648318,
author = {Choudhary, Nurendra and Huang, Edward W and Subbian, Karthik and Reddy, Chandan K.},
title = {An Interpretable Ensemble of Graph and Language Models for Improving Search Relevance in E-Commerce},
year = {2024},
isbn = {9798400701726},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589335.3648318},
doi = {10.1145/3589335.3648318},
booktitle = {Companion Proceedings of the ACM Web Conference 2024},
pages = {206–215},
numpages = {10},
keywords = {e-commerce, ensemble, graphs, language models, plug and play, query, search relevance},
location = {Singapore, Singapore},
series = {WWW '24}
}
```

## Requirements
```
absl-py==1.3.0
aiohttp==3.9.4
aiosignal==1.2.0
anyio==3.6.2
apex==0.1
argon2-cffi-bindings==21.2.0
argon2-cffi==21.3.0
asttokens==2.0.8
async-timeout==4.0.2
attrs==22.1.0
backcall==0.2.0
beautifulsoup4==4.11.1
bleach==5.0.1
cachetools==5.2.0
certifi==2024.7.4
cffi==1.15.1
charset-normalizer==2.1.1
click==8.1.3
colorcet==3.0.1
contourpy==1.0.5
cycler==0.11.0
datasets==2.6.1
debugpy==1.6.3
decorator==5.1.1
defusedxml==0.7.1
dill==0.3.5.1
docker-pycreds==0.4.0
emoji==1.7.0
entrypoints==0.4
executing==1.1.1
fastjsonschema==2.16.2
filelock==3.8.0
fire==0.4.0
fonttools==4.43.0
frozenlist==1.3.1
fsspec==2022.10.0
gitdb==4.0.9
gitpython==3.1.41
google-auth-oauthlib==0.4.6
google-auth==2.13.0
grpcio==1.53.2
huggingface-hub==0.10.1
idna==3.7
importlib-metadata==5.0.0
importlib-resources==5.10.0
ipykernel==6.16.1
ipython-genutils==0.2.0
ipython==8.10.0
jedi==0.18.1
jinja2==3.1.4
joblib==1.2.0
jsonschema==4.16.0
jupyter-client==7.4.3
jupyter-core==4.11.2
jupyter-server==2.11.2
jupyterlab-pygments==0.2.2
kiwisolver==1.4.4
lightning-transformers==0.2.3
lightning-utilities==0.3.0
markdown==3.4.1
markupsafe==2.1.1
matplotlib-inline==0.1.6
matplotlib==3.6.0
mistune==2.0.4
multidict==6.0.2
multiprocess==0.70.13
nbclassic==0.4.5
nbclient==0.7.0
nbconvert==7.2.2
nbformat==5.7.0
nest-asyncio==1.5.6
nltk==3.7
notebook-shim==0.2.0
notebook==6.5.1
numpy==1.23.4
oauthlib==3.2.2
packaging==21.3
pandas==1.5.1
pandocfilters==1.5.0
param==1.12.2
parso==0.8.3
pathtools==0.1.2
pexpect==4.8.0
pickleshare==0.7.5
pillow==10.3.0
pip==23.3
pkg-resources==0.0.0
pkgutil-resolve-name==1.3.10
prometheus-client==0.15.0
promise==2.3
prompt-toolkit==3.0.31
protobuf==3.19.6
psutil==5.9.3
ptyprocess==0.7.0
pure-eval==0.2.2
pyarrow==14.0.1
pyasn1-modules==0.2.8
pyasn1==0.4.8
pycparser==2.21
pyct==0.4.8
pydeprecate==0.3.2
pygments==2.15.0
pyparsing==3.0.9
pyrsistent==0.18.1
python-dateutil==2.8.2
pytorch-lightning==1.7.7
pytz==2022.5
pyyaml==6.0
pyzmq==24.0.1
regex==2022.9.13
requests-oauthlib==1.3.1
requests==2.32.2
responses==0.18.0
rsa==4.9
sacremoses==0.0.53
seaborn==0.12.1
send2trash==1.8.0
sentencepiece==0.1.97
sentry-sdk==1.14.0
setproctitle==1.3.2
setuptools==65.5.1
shortuuid==1.0.9
six==1.16.0
smmap==5.0.0
sniffio==1.3.0
soupsieve==2.3.2.post1
stack-data==0.5.1
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorboard==2.10.1
termcolor==2.0.1
terminado==0.16.0
tinycss2==1.2.1
tokenizers==0.13.1
torch==1.13.1
torchmetrics==0.10.0
tornado==6.4.1
tqdm==4.64.1
traitlets==5.5.0
transformer-utils==0.1.1
transformers==4.38.0
typing-extensions==4.4.0
urllib3==1.26.19
wandb==0.13.4
wcwidth==0.2.5
webencodings==0.5.1
websocket-client==1.4.1
werkzeug==3.0.3
wheel==0.38.0
xxhash==3.1.0
yarl==1.8.1
zipp==3.19.1
```

