import gc
from graph_config import RELATION_DICT, PROCESSED_FILENAMES
import logging
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ESCIGraphDataset(InMemoryDataset):
    """
    Dataset Class to load behavioral signals from different graphs
    and add them to the primary prediction ESCI graph. 
    """
    def __init__(self, root, relation, transform=None, 
                pre_transform=None, pre_filter=None):
        """
        Dataset Constructor class:
        root: str: main directory containing the main dataset information. 
        relation: str: indicates the behavior signal to be used, could be
                       impressions/clicks/adds/purchases/consumes/all/all_attr
        """ 
        super().__init__(root, transform, pre_transform, pre_filter)   
        if relation == "impressions":
            self.data = torch.load(self.processed_paths[0])
        elif relation == "clicks":
            self.data = torch.load(self.processed_paths[1])
        elif relation == "adds":
            self.data = torch.load(self.processed_paths[2])
        elif relation == "purchases":
            self.data = torch.load(self.processed_paths[3])
        elif relation == "consumes":
            self.data = torch.load(self.processed_paths[4])
        elif relation == "all":
            self.data = torch.load(self.processed_paths[5])
        elif relation == "all_attr":
            self.data = torch.load(self.processed_paths[6])
        else:
            raise ValueError(f"{relation} not supported. 
                            Supported relations are {list(RELATION_DICT.keys())}")
        with open(self.processed_paths[7],"rb") as map_file:
            self.node_id_map = pickle.load(map_file)
        with open(self.processed_paths[8],"rb") as edge_file:
            self.edge_label_set = pickle.load(edge_file)
    
    @property
    def raw_file_names(self):
        """
        Returns name of the raw data files.
        If these files do not exist, rest of the procedure will fail.
        """ 
        self.relation_dict = RELATION_DICT
        relation_list = [f"relation_{relation}" for relation in self.relation_dict.values()]
        relation_list = relation_list + ["esci"]
        train_filenames = [f"train/{filename}" for filename in relation_list]
        val_filenames = [f"val/{filename}" for filename in relation_list]
        test_filenames = [f"test/{filename}" for filename in relation_list]
        return ["features.npy"]+train_filenames+val_filenames+test_filenames+["node_id_map.pkl"]

    @property
    def processed_file_names(self):
        """
        Returns name of the processed data files.
        Needs to have valid raw files.
        If these files do not exist, self.process() will be called.
        """
        
        return PROCESSED_FILENAMES
    
    def load_node_feats(self, node_feat_path):
        """
        Method to load preprocessed node features
        """
        return np.load(node_feat_path)

    def load_esci(self, train_esci_path, val_esci_path, test_esci_path):
        """
        Method to load ESCI dataset as (edge,label) pairs for train, val and test splits.
        """
        train_esci = pd.read_parquet(train_esci_path)
        input_train_labels = torch.tensor(pd.get_dummies(train_esci["label"]).values).tile((2,1))
        train_src = torch.tensor(list(train_esci["keyword"])+list(train_esci["DocumentId"]),dtype=int)
        train_dst = torch.tensor(list(train_esci["DocumentId"])+list(train_esci["keyword"]),dtype=int)
        input_train_edges = torch.stack([train_src,train_dst])
        
        val_esci = pd.read_parquet(val_esci_path)
        input_val_labels = torch.tensor(pd.get_dummies(val_esci["label"]).values).tile((2,1))
        val_src = torch.tensor(list(val_esci["keyword"])+list(val_esci["DocumentId"]),dtype=int)
        val_dst = torch.tensor(list(val_esci["DocumentId"])+list(val_esci["keyword"]),dtype=int)
        input_val_edges = torch.stack([val_src,val_dst])
        
        test_esci = pd.read_parquet(test_esci_path)
        input_test_labels = torch.tensor(pd.get_dummies(test_esci["label"]).values).tile((2,1))
        test_src = torch.tensor(list(test_esci["keyword"])+list(test_esci["DocumentId"]),dtype=int)
        test_dst = torch.tensor(list(test_esci["DocumentId"])+list(test_esci["keyword"]),dtype=int)
        input_test_edges = torch.stack([test_src,test_dst])

        return input_train_edges, input_train_labels, input_val_edges, input_val_labels, input_test_edges, input_test_labels

    def make_datagraph(self, node_feat_path, edge_paths, 
                        train_esci_path, val_esci_path, test_esci_path,
                        with_attr = 0):
        """
        Method to structure raw graphs into "Data" class of torch_geometric.
        """
        node_feat = self.load_node_feats(node_feat_path)
        node_feat = torch.Tensor(node_feat)
        srcs, dsts = [],[]
        if with_attr: 
            edge_attrs = []
        for ind, edge_path in enumerate(edge_paths):
            edges = pd.read_parquet(edge_path)
            srcs.append(torch.tensor(list(edges["keyword"])+list(edges["DocumentId"]),dtype=int))
            dsts.append(torch.tensor(list(edges["DocumentId"])+list(edges["keyword"]),dtype=int))
            # Adding a one-hot encoding of the edge attributes to the graph dataset.
            # Graph is undirected, hence number of edges is len(edges["keyword"])*2
            num_attrs = len(edges["keyword"])*2
            # Zero initialization of each relation
            attr = [0]*len(RELATION_DICT)
            # Edge Path consists of train, test and val for each relation
            # Hence, ind//3 gives the edge attribute for each set.
            attr[ind//3] = 1
            if with_attr:
                edge_attrs.append(torch.tensor([attr]*num_attrs,dtype=float))
        edge_index = torch.stack([torch.cat(srcs),torch.cat(dsts)])
        if with_attr:
            edge_attrs = torch.cat(edge_attrs)
        if with_attr:
            data = Data(x=node_feat,edge_index=edge_index, edge_attr=edge_attrs)
        else: data = Data(x=node_feat,edge_index=edge_index)
        return data


    def process(self):
        """
        Processing method to convert raw files into processed files.
        self.raw_paths stores the return from self.raw_file_names()
        """
        logging.info("Starting the Process phase.")
        node_feat_path = self.raw_paths[0]
        train_impressions_path = self.raw_paths[1]
        train_clicks_path = self.raw_paths[2]
        train_adds_path = self.raw_paths[3]
        train_purchases_path = self.raw_paths[4]
        train_consumes_path = self.raw_paths[5]
        train_esci_path = self.raw_paths[6]

        val_impressions_path = self.raw_paths[7]
        val_clicks_path = self.raw_paths[8]
        val_adds_path = self.raw_paths[9]
        val_purchases_path = self.raw_paths[10]
        val_consumes_path = self.raw_paths[11]
        val_esci_path = self.raw_paths[12]

        test_impressions_path = self.raw_paths[13]
        test_clicks_path = self.raw_paths[14]
        test_adds_path = self.raw_paths[15]
        test_purchases_path = self.raw_paths[16]
        test_consumes_path = self.raw_paths[17]
        test_esci_path = self.raw_paths[18]

        logging.info("Loading from the graph paths.")

        logging.info("Loading data of relation impressions.")
        data_impressions = self.make_datagraph(node_feat_path,[train_impressions_path,val_impressions_path,test_impressions_path],
                                                train_esci_path, val_esci_path, test_esci_path)
        torch.save(data_impressions,os.path.join(self.processed_dir,"data_impressions.pt"))

        logging.info("Loading data of relation clicks.")
        data_clicks = self.make_datagraph(node_feat_path,[train_clicks_path,val_clicks_path,test_clicks_path],
                                            train_esci_path, val_esci_path, test_esci_path)
        torch.save(data_clicks,os.path.join(self.processed_dir,"data_clicks.pt"))

        logging.info("Loading data of relation adds.")
        data_adds = self.make_datagraph(node_feat_path,[train_adds_path,val_adds_path,test_adds_path],
                                        train_esci_path, val_esci_path, test_esci_path)
        torch.save(data_adds,os.path.join(self.processed_dir,"data_adds.pt"))

        logging.info("Loading data of relation purchases.")
        data_purchases = self.make_datagraph(node_feat_path,[train_purchases_path,val_purchases_path,test_purchases_path],
                                            train_esci_path, val_esci_path, test_esci_path)
        torch.save(data_purchases,os.path.join(self.processed_dir,"data_purchases.pt"))
        
        logging.info("Loading data of relation consumes.")
        data_consumes = self.make_datagraph(node_feat_path,[train_consumes_path,val_consumes_path,test_consumes_path],
                                            train_esci_path, val_esci_path, test_esci_path)
        torch.save(data_consumes,os.path.join(self.processed_dir,"data_consumes.pt"))

        logging.info("Loading data of all relations.")
        data_all = self.make_datagraph(node_feat_path,[train_impressions_path,val_impressions_path,test_impressions_path,
                                                       train_clicks_path,val_clicks_path,test_clicks_path,
                                                       train_adds_path,val_adds_path,test_adds_path,
                                                       train_purchases_path,val_purchases_path,test_purchases_path,
                                                       train_consumes_path,val_consumes_path,test_consumes_path],
                                        train_esci_path, val_esci_path, test_esci_path)
        torch.save(data_all,os.path.join(self.processed_dir,"data_all.pt"))

        logging.info("Loading data of all relations with attributes.")
        data_all_attr = self.make_datagraph(node_feat_path,[train_impressions_path,val_impressions_path,test_impressions_path,
                                                       train_clicks_path,val_clicks_path,test_clicks_path,
                                                       train_adds_path,val_adds_path,test_adds_path,
                                                       train_purchases_path,val_purchases_path,test_purchases_path,
                                                       train_consumes_path,val_consumes_path,test_consumes_path],
                                        train_esci_path, val_esci_path, test_esci_path, with_attr=1)    
        torch.save(data_all_attr,os.path.join(self.processed_dir,"data_all_attr.pt"))
        
        shutil.copyfile(os.path.join(self.raw_paths[-1]), os.path.join(self.processed_dir,"node_id_map.pkl"))
        input_train_edges, input_train_labels, input_val_edges, input_val_labels, input_test_edges, input_test_labels = self.load_esci(train_esci_path, val_esci_path, test_esci_path)
        edge_label_set = {}
        edge_label_set["input_train_edges"] = input_train_edges
        edge_label_set["input_train_labels"] = input_train_labels
        edge_label_set["input_val_edges"] = input_val_edges
        edge_label_set["input_val_labels"] = input_val_labels
        edge_label_set["input_test_edges"] = input_test_edges
        edge_label_set["input_test_labels"] = input_test_labels
        with open(os.path.join(self.processed_dir,"edge_label_set.pkl"),"wb") as edge_file:
            pickle.dump(edge_label_set, edge_file)


if __name__=="__main__":
    for locale in ["es","jp","us"]:
        logging.info(f"Processing {locale} dataset.")
        filedir = f"../esci_datasets/esci_graph_datasets/{locale}/"
        data = ESCIGraphDataset(root=filedir,relation="all_attr")