import os, torch, sys
import psutil

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

MODEL_NAME =  'bert-base-uncased'
MODEL_NAME =  'distilbert-base-uncased'
# MODEL_NAME =  'sshleifer/tiny-distilbert-base-cased'

data_folder = '/home/anna/seq_classify/data/aclImdb'
train_fname = data_folder + '/imdb_train_df.csv'
test_fname = data_folder + '/imdb_test_df.csv'

MAX_SEQ_LEN = 512
NUM_EPOCHS = 10
BATCH_SIZE = 30
LR = 0.00001
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
BERT_LAYER_FREEZE = False

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

#when using xlarge vs 16x large AWS m/c
MULTIGPU = True if torch.cuda.device_count() > 1 else False

CONTEXT_VECTOR_SIZE = 1024 if 'large' in MODEL_NAME else 768
IS_LOWER = True if 'uncased' in MODEL_NAME else False

SAVE_EVERY = 2
SAVE_MODEL_FNAME = f"{MODEL_NAME.upper().replace('/','_')}_e_.pt"


