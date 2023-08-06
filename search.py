try:
    from src import build_index
    from src import utils
    from src import preprocessor
    from src import embeddings
    from src import config
except:
    import build_index
    import utils
    import preprocessor
    import embeddings
    import config

import os
import datetime
import uuid
import pandas as pd
import numpy as np

logger = utils.logger

lookup = utils.load_article_lookup()

def index_to_content(index:int, input_data_type=config.TRAIN_DATA_INPUT_TYPES)->str:
    """
    get the content of the article with the given index
    :param index: int, index of the article
    :param input_data_type: list, list of input data types
    :return: str, content of the article
    """
    assert isinstance(index, int), f"index: {index} {type(index)}"
    assert isinstance(input_data_type, list)
    assert isinstance(input_data_type[0], str)

    embs1 = []
    embs2 = []
    if input_data_type==["title"]:
        title = lookup[index]["title"]
        return embeddings.get_embeddings_from_lemmatized_sentences([preprocessor.preprocess_text(title)])
    elif input_data_type==["text"]:
        text = lookup[index]["text"]
        return embeddings.get_embeddings_from_lemmatized_sentences([preprocessor.preprocess_text(text)])
    else:
        assert "title" in input_data_type
        assert "text" in input_data_type
        title = lookup[index]["title"]
        text = lookup[index]["text"]
        embs1 = embeddings.get_embeddings_from_lemmatized_sentences([preprocessor.preprocess_text(title)])
        embs2 = embeddings.get_embeddings_from_lemmatized_sentences([preprocessor.preprocess_text(text)])
        return embs1 + embs2

def compute_relevance_score(v1:np.ndarray, v2:np.ndarray)->float:
    """
    Compute cosine similarity between two vectors
    :param v1: np.ndarray
    :param v2: np.ndarray
    :return: float
    """
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    assert v1.shape == v2.shape, f"v1.shape: {v1.shape}, v2.shape: {v2.shape}"
    assert v1.shape[0] > 0
    return round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), config.RELEVANCE_SCORE_ROUNDING)
    

def search(index, queries:list, k:int, ids_mapper:dict, sections_stats:dict)->list:
    """
    search the index for the k most similar articles to the queries
    :param queries: list, list of strings
    :param k: int, number of articles to return
    :return: list, list of tuples (article_id, score)
    """
    assert isinstance(queries, list)
    assert isinstance(queries[0], str)
    assert isinstance(k, int)
    assert k > 0

    logger.info(f"Searching index for {len(queries)} queries...")

    _k = max(k, int(sections_stats["mean"] + sections_stats["std"])*k)
    _queries = list(map(lambda x: preprocessor.preprocess_text(x), queries))
    embs = embeddings.get_embeddings_from_lemmatized_sentences(_queries)
    _results = index.search(embs, _k, ids_mapper["section_id_to_article_id"])
    
    results = []
    for i in range(len(_results)):
        logger.info("Proceesing query:" + queries[i])
        result = {
            "query": queries[i]
        }
        result["results"] = []
        skip = set()
        for j in range(len(_results[i])):
            if len(result["results"])>=k:
                break
            
            article_id = _results[i][j][0]
            if article_id in skip:
                logger.info(f"Duplicate result - Skipping: {article_id}")
                continue
            res_emb = index_to_content(article_id, config.TRAIN_DATA_INPUT_TYPES)[0]
            
            relevance_score = compute_relevance_score(np.asarray(embs[i]).T, np.asarray(res_emb))
            if relevance_score < config.RELEVANCE_SCORE_THRESHOLD:
                logger.info(f"Relevance score below threshold - Skipping: {article_id}")
                continue
            
            result["results"].append({
                "article_id": article_id,
                "score": relevance_score,
                "title": lookup[int(article_id)]["title"],
                "category": lookup[int(article_id)]["category"],
                "subcategory": lookup[int(article_id)]["subcategory"]
            })
            skip.add(_results[i][j][0])
        results.append(result)
    
    logger.info(f"Search complete.")
    return results


if __name__=="__main__":
    from pprint import pprint
    index = build_index.init_index(index_type=config.SEARCH_INDEX_TYPE)
    index.load(os.path.join(config.MODELS_DIR, config.SEARCH_INDEX))
    ids_mapper = utils.load_json(os.path.join(config.MODELS_DIR, f"{config.TEXT_SECTION_TYPE}_{'_'.join(config.TRAIN_DATA_INPUT_TYPES)}_ids.json"))
    sections_stats = utils.load_json(os.path.join(config.MODELS_DIR, f"{config.TEXT_SECTION_TYPE}_{'_'.join(config.TRAIN_DATA_INPUT_TYPES)}_stats.json"))["sections_by_article"]

    # queries = [lookup[i]["title"] for i in (0, 1)]
    queries = [
        "Carbon emission",
        "debt reduction among companies",
        "lack of medical devices in hospitals",
        "covid-19 vaccine development",
        "women entrepreneuship in india",
    ]
    print(queries)
    results = search(index, queries, 5, ids_mapper, sections_stats)
    utils.save_json(results, os.path.join(config.RESULTS_DIR, f'search_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'))

    

