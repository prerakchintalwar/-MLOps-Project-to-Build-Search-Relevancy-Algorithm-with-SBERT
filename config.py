### --> I. Set the environment to "dev" or "prod"
ENV = "prod"

DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

### --> II. The following are used for training
SAMPLE_SIZE = None
REUSE_PREPROCESSED_DATA = True
REUSE_PREGENERATED_EMBEDDINGS = True


CONNECTION_STRING = "enter-your-connection-string"

# chose to section text by sentences or paragraphs
# It can take values from ["sentence", "paragraph", None]
TEXT_SECTION_TYPE = None
assert TEXT_SECTION_TYPE in ["sentence", "paragraph", None], "TEXT_SECTION_TYPE can only take values from ['sentence', 'paragraph', None]"

# define the input types for the training data. 
# It can be either ["title"] or ["text"] or both ["title", "text"]
TRAIN_DATA_INPUT_TYPES = ["title"]
TRAIN_DATA_INPUT_TYPES.sort(reverse=True)
SEARCH_INDEX_TYPE = "annoy"
assert SEARCH_INDEX_TYPE in ["annoy"], "SEARCH_INDEX_TYPE can only take values from ['annoy', 'faiss']"
ANNOY_N_TREES = 50
ANNOY_METRIC = "euclidean" # "manhattan", "angular"
SEARCH_INDEX = f"{TEXT_SECTION_TYPE}_" + "_".join(TRAIN_DATA_INPUT_TYPES) + f".{SEARCH_INDEX_TYPE}"
if TEXT_SECTION_TYPE=="sentence":
    assert "title" not in TRAIN_DATA_INPUT_TYPES, "title is not supported for sentence section type"

# Define model type for SentenceTransformer
# It can take values from https://www.sbert.net/docs/pretrained_models.html
# 'all-MiniLM-L6-v2' is the smallest and fastest model with the good performance
# 'distilbert-base-nli-stsb-mean-tokens' has best performance
# Note: The models 'distilbert-base-nli-stsb-mean-tokens' and 'bert-base-nli-mean-tokens' are both 
# fine-tuned versions of the popular transformer models DistilBERT and BERT respectively. 
# The main difference between the two is that DistilBERT is a smaller, faster, and lighter version of BERT, 
# while still preserving a lot of its performance.
# - 'distilbert-base-nli-stsb-mean-tokens' is a fine-tuned DistilBERT model on the Natural Language Inference (NLI) and 
# the Sentences Transformation (STS-B) benchmark tasks. It has been trained to generate sentence embeddings by 
# taking the mean of the token embeddings.
# - 'bert-base-nli-mean-tokens' is a fine-tuned BERT model on the Natural Language Inference (NLI) task. Similar 
# to the above model, it has been trained to generate sentence embeddings by taking the mean of the token embeddings.
SENTENCE_TRANSFORMER_MODEL_TYPE = "all-MiniLM-L6-v2"
if SENTENCE_TRANSFORMER_MODEL_TYPE=="all-MiniLM-L6-v2":
    ANNOY_SIZE = 384
elif SENTENCE_TRANSFORMER_MODEL_TYPE=="distilbert-base-nli-stsb-mean-tokens":
    ANNOY_SIZE = 768
elif SENTENCE_TRANSFORMER_MODEL_TYPE=="bert-base-nli-mean-tokens":
    ANNOY_SIZE = 768
else:
    raise Exception("Invalid SENTENCE_TRANSFORMER_MODEL_TYPE")


### --> III. The following are used for inference
RELEVANCE_SCORE_ROUNDING = 2
RELEVANCE_SCORE_THRESHOLD = 0.4







