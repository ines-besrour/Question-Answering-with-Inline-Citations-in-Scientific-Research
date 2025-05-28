import torch

EMBEDDING_MODEL = "intfloat/e5-large-v2"
MODEL_FORMAT = "sentence_transformers"
EMBEDDING_DIM = 1024
USE_GPU = torch.cuda.is_available() 

# Configuration paths
DATA_DIR = "/data/horse/ws/inbe405h-unarxive/processed_unarxive_extended_data"
DOC_INDEX_DIR = "/data/horse/ws/inbe405h-unarxive/faiss_index"
