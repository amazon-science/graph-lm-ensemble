DATASET_FILES = {
            "us": "", #Path to US dataset 
            "es": "", #Path to ES dataset
            "jp": "" #Path to JP dataset
        }
GNN_MODELS = ["SAGE","GCN","GAT"]
PROCESSED_FILENAMES = ['data_impressions.pt', 'data_clicks.pt',
                       'data_adds.pt', 'data_purchases.pt',
                       'data_consumes.pt','data_all.pt', 
                       'data_all_attr.pt','node_id_map.pkl', 
                       'edge_label_set.pkl']
RELATION_DICT = {
            0: "impressions", 
            1: "clicks", 
            2: "adds", 
            3:"purchases", 
            4: "consumes"
        }
RELATIONS = ["impressions","adds","clicks","purchases","consumes","all","all_attr"]
NUM_LABELS = 4
NUM_FEATURES = 768
