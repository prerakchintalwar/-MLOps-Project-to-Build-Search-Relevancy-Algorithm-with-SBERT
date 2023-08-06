import numpy as np
import faiss
import annoy
import os
from functools import reduce

try:
    from src import utils
    from src import config
except:
    import utils
    import config

logger = utils.logger

class AnnoyIndexer:
    """
    Class to build and search an annoy index
    """
    def __init__(self):
        self.index = None

    def build(self, embs_obj:dict, index_path:str):
        """
        build a similarity index for the embeddings in embs_obj using annoy 
        :param embs_obj: dict, keys are article ids and values are embeddings
        :param index_path: str, path to save the index
        """
        assert isinstance(embs_obj, dict)
        assert isinstance(index_path, str)
        indices = list(embs_obj.keys())
        embeddings = list(embs_obj.values())
        
        assert len(embeddings[0])==config.ANNOY_SIZE, f"Embedding size {len(embeddings[0])} does not match ANNOY_SIZE {config.ANNOY_SIZE}"
        
        self.index = annoy.AnnoyIndex(config.ANNOY_SIZE, config.ANNOY_METRIC)
        for i, embedding in zip(indices, embeddings):
            self.index.add_item(int(i), embedding)
        self.index.build(config.ANNOY_N_TREES)
        self.save(index_path)

    def search(self, query_embeddings:list, k:int, ids_lookup:dict=None)->list:
        """
        search for the k nearest neighbors of the query embeddings
        :param query_embeddings: list, list of query embeddings
        :param k: int, number of nearest neighbors to return
        :param ids_lookup: dict, keys are article ids and values are embeddings
        :return: list, list of tuples of the form (id, distance)
        """
        assert isinstance(query_embeddings, list)
        assert isinstance(k, int)
        assert isinstance(ids_lookup, dict) or ids_lookup is None

        results = []
        for query in query_embeddings:
            result = self.index.get_nns_by_vector(query, k, include_distances=True)
            ids = list(map(lambda x: ids_lookup[str(x)], result[0]))
            if config.ANNOY_METRIC == "angular":
                res = sorted(set(zip(ids, result[1])), key=lambda x: x[1], reverse=True)
            elif config.ANNOY_METRIC == "euclidean":
                res = sorted(set(zip(ids, result[1])), key=lambda x: x[1], reverse=False)
            else:
                raise ValueError(f"Unknown metric: {config.ANNOY_METRIC}")
            results.append(res)
        return results

    def save(self, index_path:str):
        """
        save the index to disk
        :param index_path: str, path to save the index
        """
        assert isinstance(index_path, str)
        self.index.save(index_path)

    def load(self, index_path:str):
        """
        load the index from disk
        :param index_path: str, path to load the index from
        """
        assert isinstance(index_path, str)
        self.index = annoy.AnnoyIndex(config.ANNOY_SIZE, config.ANNOY_METRIC)
        self.index.load(index_path)

class FaissIndex:
    """
    Class to build and search a faiss index
    """
    def __init__(self):
        self.index = None

    def build(self, embs_obj:dict, index_path:str):
        """
        build a similarity index for the embeddings in embs_obj using faiss 
        :param embs_obj: dict, keys are article ids and values are embeddings
        :param index_path: str, path to save the index
        """
        assert isinstance(embs_obj, dict)
        assert isinstance(index_path, str)
        
        logger.info(f"Building index for {len(embs_obj)} embeddings...")


        embs = []
        section_ids = []
        for k, v in embs_obj.items():
            embs.append(v)
            section_ids.append(k)
        embs = np.asarray(np.array(embs).astype(np.float32))
        # self.index = faiss.IndexIDMap(faiss.IndexFlatIP(embs.shape[1]))
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(embs.shape[1]))
        self.index.add_with_ids(embs, np.array(section_ids).astype(np.int64))

        logger.info(f"Index built with {self.index.ntotal} embeddings and dimension {self.index.d}...")
        logger.info(f"Saving index to {index_path}...")
        self.save(index_path)
    
    def search(self, query_emb:list, k:int, ids_lookup:dict=None)->list:
        """
        search the index for the k most similar articles to the query embedding
        :param query_emb: list of lists, query embedding
        :param k: int, number of articles to return
        :return: list, list of tuples (article_id, score)
        """
        assert isinstance(k, int)
        assert isinstance(query_emb, list)
        assert isinstance(query_emb[0], list)
        assert isinstance(query_emb[0][0], float)
        assert len(query_emb[0]) == self.index.d, f"Embedding dimension {len(query_emb[0])} does not match index dimension {self.index.d}"
        assert k > 0
        assert isinstance(ids_lookup, dict) or ids_lookup is None

        query_emb = np.asarray(np.array(query_emb).astype(np.float32))

        D, I = self.index.search(query_emb, k)
        results = list()
        for i in range(len(I)):
            ids = list(map(lambda x: ids_lookup[str(x)], I[i].tolist()))
            res = sorted(set(zip(ids, D[i].tolist())), key=lambda x: x[1])
            results.append(res)
        return results
    
    def save(self, index_path:str)->None:
        """
        save the index to disk
        :param index_path: str, path to save the index
        """
        assert isinstance(index_path, str)
        assert self.index is not None
        faiss.write_index(self.index, index_path)
        logger.info(f"Index saved to {index_path}")
    
    def load(self, index_path:str)->None:
        """
        load the index from disk
        :param index_path: str, path to load the index from
        """
        assert isinstance(index_path, str)
        assert os.path.exists(index_path)
        self.index = faiss.read_index(index_path)
        logger.info(f"Index loaded from {index_path}")

def init_index(index_type:str)->object:
    """
    initialize an index object based on the index type
    :param index_type: str, type of index to initialize
    :return: object, index object
    """
    assert isinstance(index_type, str)
    assert index_type in ["annoy", "faiss"]
    if index_type == "annoy":
        return AnnoyIndexer()
    elif index_type == "faiss":
        return FaissIndex()

if __name__ == "__main__":
    embs = utils.load_json(os.path.join(config.DATA_DIR, "processed", f"{config.TEXT_SECTION_TYPE}_{'_'.join(config.TRAIN_DATA_INPUT_TYPES)}_embeddings.json"))
    index = init_index(config.SEARCH_INDEX_TYPE)
    index.build(embs, os.path.join(config.MODELS_DIR, config.SEARCH_INDEX))
    index.load(os.path.join(config.MODELS_DIR, config.SEARCH_INDEX))
    
    ids_mapper = utils.load_json(os.path.join(config.MODELS_DIR, f"{config.TEXT_SECTION_TYPE}_" + "_".join(config.TRAIN_DATA_INPUT_TYPES) + "_ids.json"))
    query_emb = [embs["0"], embs["1"], embs["20"]]
    results = index.search(query_emb, 5, ids_lookup=ids_mapper["section_id_to_article_id"])
    print(results)