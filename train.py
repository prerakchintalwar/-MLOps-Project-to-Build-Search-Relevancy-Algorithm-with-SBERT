try:
    from src import config
    from src import utils
    from src import preprocessor
    from src import build_index
    from src import embeddings
except:
    import config
    import utils
    import preprocessor
    import build_index
    import embeddings

import pandas as pd
import os
import logging
from functools import reduce


logger = utils.logger

def execute():
    logger.info("Executing the training pipeline...")
    if config.REUSE_PREPROCESSED_DATA:
        assert os.path.exists(os.path.join(config.DATA_DIR, "processed", f"{config.TEXT_SECTION_TYPE}_{'_'.join(config.TRAIN_DATA_INPUT_TYPES)}_processed.csv")), f'Preprocessed data not found! Please set REUSE_PREPROCESSED_DATA to False in config.py and run the pipeline again.'
        logger.info("Reusing processed data...")
        r = dict()
        r["path_to_processed_text"] = os.path.join(config.DATA_DIR, "processed", f"{config.TEXT_SECTION_TYPE}_{'_'.join(config.TRAIN_DATA_INPUT_TYPES)}_processed.csv")
        r["path_to_article_ids"] = os.path.join(config.MODELS_DIR, f"{config.TEXT_SECTION_TYPE}_" + "_".join(config.TRAIN_DATA_INPUT_TYPES) + "_ids.json")
    else:
        logger.info("Collecting and preprocessing data ...")
        output_filename = utils.get_raw_data_from_aws_mongo()
        r = preprocessor.preprocess_data(output_filename, section_by=config.TEXT_SECTION_TYPE, input_types=config.TRAIN_DATA_INPUT_TYPES, sample_size=config.SAMPLE_SIZE)
    
    pdf = pd.read_csv(r["path_to_processed_text"])
    assert "section_id" in pdf.columns
    assert "text" in pdf.columns
    assert "article_id" in pdf.columns

    logger.info("Building title index ...")
    if not config.REUSE_PREGENERATED_EMBEDDINGS:
        data = dict(
            zip(
                pdf["section_id"].values.tolist(), 
                embeddings.get_embeddings_from_lemmatized_sentences(pdf["text"].values.tolist())
            )
        )
        utils.save_json(data, os.path.join(config.DATA_DIR, "processed", f"{config.TEXT_SECTION_TYPE}_" + "_".join(config.TRAIN_DATA_INPUT_TYPES) + "_embeddings.json"))
    else:
        assert os.path.exists(os.path.join(config.DATA_DIR, "processed", f"{config.TEXT_SECTION_TYPE}_" + "_".join(config.TRAIN_DATA_INPUT_TYPES) + "_embeddings.json")), "Embeddings not found. Please set REUSE_PREGENERATED_EMBEDDINGS to False in config.py and run the pipeline again."
    
    _embeddings = utils.load_json(os.path.join(config.DATA_DIR, "processed", f"{config.TEXT_SECTION_TYPE}_" + "_".join(config.TRAIN_DATA_INPUT_TYPES) + "_embeddings.json"))
    index = build_index.init_index(index_type=config.SEARCH_INDEX_TYPE)
    index.build_index(_embeddings, os.path.join(config.MODELS_DIR, config.SEARCH_INDEX))
    

    logger.info("Training pipeline completed successfully!")
    
if __name__ == "__main__":
    execute()
    
    