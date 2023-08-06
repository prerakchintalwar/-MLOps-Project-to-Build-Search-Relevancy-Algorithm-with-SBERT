from sentence_transformers import SentenceTransformer
try:
    from src import config
    from src import utils
except:
    import config
    import utils


logger = utils.logger

model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL_TYPE)

def get_embeddings_from_lemmatized_sentences(sentences:list, mode:str="training", batch_size:int=32)->list:
    """
    Get embeddings from lemmatized sentences
    :param sentences: list of lemmatized sentences
    :param mode: str, can be "training" or "inference"
    :param batch_size: int
    :param num_workers: int
    :return: list of embeddings
    """
    assert isinstance(sentences, list)
    assert isinstance(mode, str)
    assert isinstance(batch_size, int)

    assert mode in ("training", "inference")
    # logger.info(f"Getting embeddings from lemmatized sentences in {mode} mode...")
    embeddings = model.encode(sentences, show_progress_bar=False, device="cpu", batch_size=batch_size)
    return embeddings.tolist()

if __name__=="__main__":
    sentences = [ "This is a sentence", "This is another sentence", "this is a very kilometric sentence to check if the embeddings are working properly"]
    sentence_embeddings = get_embeddings_from_lemmatized_sentences(sentences)
    print(sentence_embeddings)
    print(type(sentence_embeddings))
