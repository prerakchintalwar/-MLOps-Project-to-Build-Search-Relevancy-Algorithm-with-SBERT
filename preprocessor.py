
import spacy
import re
import os
import tqdm
import pandas as pd
from functools import reduce
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as nltk_stop_words

try:
    from src import config
    from src import utils
except:
    import config
    import utils

logger = utils.logger

STOP_WORDS = set(spacy_stop_words).union(set(nltk_stop_words.words("english")))
SPACY_TOKENIZER = spacy.load('en_core_web_sm')

def separate_capitalilzed_words(text:str)->str:
    """
    Introduce space before each capital letter
    Example: "ThisIsAWord" -> "This Is A Word"
    :param text: str
    :return: str
    """
    assert isinstance(text, str)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text

def lower_case(text:str)->str:
    """
    Lower case text
    :param text: str
    :return: str
    """
    assert isinstance(text, str)
    text = text.lower()
    return text

def fix_white_space(text:str)->str:
    """
    Fix white space
    :param text: str
    :return: str
    """
    assert isinstance(text, str)
    text = "\n".join(" ".join(text.split()).split("\n"))
    return text

def text_to_paragraphs(text:str)->list:
    """
    Convert text to paragraphs
    :param text: str
    :return: list
    """
    if not isinstance(text, str):
        return []
    paragraphs = list(filter(lambda x: x!="", text.split("\n")))
    return paragraphs

def text_to_sentences(text:str)->list:
    """
    Convert text to sentences
    :param text: str
    :return: list
    """
    if not isinstance(text, str):
        return []
    sentences = SPACY_TOKENIZER(text).sents
    sentences = [str(sentence) for sentence in sentences]
    return sentences

def remove_non_alphanumeric(text:str)->str:
    """
    Remove everything except alphanumeric characters and spaces
    :param text: str
    :return: str
    """
    assert isinstance(text, str)
    pattern = '[^a-zA-Z0-9%\ \n]+'
    return re.sub(pattern, '', text)

def remove_digits(text:str)->str:
    """
    Remove digits
    :param text: str
    :return: str
    """
    assert isinstance(text, str)
    text = re.sub(r'\d+', '', text)
    return text

def remove_stop_words(text:str)->str:
    """
    Remove stop words
    :param text: str
    :return: str
    """
    assert isinstance(text, str)
    text = " ".join([word for word in text.lower().split() if word not in STOP_WORDS])
    return text

def spacy_tokenize(text:str)->list:
    """
    Tokenize text
    :param text: str
    :return: list
    """
    assert isinstance(text, str)
    return " ".join([token.text for token in SPACY_TOKENIZER(text)])

def spacy_lemmatize(text)->str:
    """
    Lemmatize text to find the root of the word
    :param text: str
    :return: str
    """
    assert isinstance(text, str)
    text = " ".join([token.lemma_ for token in SPACY_TOKENIZER(text)])
    return text

def preprocess_text(text:str)->str:
    """
    Preprocess text
    :param text: str
    :return: str
    """
    text = str(text)
    text = separate_capitalilzed_words(text)
    text = lower_case(text)
    text = remove_non_alphanumeric(text)
    text = remove_digits(text)
    text = fix_white_space(text)
    text = remove_stop_words(text)
    text = spacy_tokenize(text)
    text = spacy_lemmatize(text)
    return text

def dedup_data(df:pd.DataFrame):
    """
    Deduplicate data.
    News article might be duplicated in the dataset after preprocessing but will have different article_id
    This function will return a dataframe with unique title and text together with a dictionary that maps 
    an article_id (returned in the data frame) to a list of article_ids
    :param df: pd.DataFrame
    :return: (pd.DataFrame, dict, dict)
    """
    assert isinstance(df, pd.DataFrame)
    assert 'article_id' in df.columns
    assert "section_id" in df.columns
    assert 'text' in df.columns
    logger.info("Deduplicating data ...")

    gdf1 = df[['article_id', 'text']].groupby(['text']).agg({'article_id': 'unique'})
    article_id_to_article_id = dict(map(lambda x: (int(x[0]), x.tolist()), gdf1['article_id'].values.tolist()))
    section_id_to_article_id = dict(map(lambda x: (int(x[0]), x[1]), df[['section_id', 'article_id']].drop_duplicates().values.tolist()))
    df = df[df.article_id.isin(list(article_id_to_article_id.keys()))][["article_id", "section_id", "text"]].sort_values("article_id")
    return df, article_id_to_article_id, section_id_to_article_id

def preprocess_article(article_id:int, title:str, text:str, section_by:str="paragraph", input_types=["title", "text"])->list:
    """
    Split article into sections.
    title and text are to be preprocessed.
    text will be split into sections based on section_by.
    Only sections with length greater than 2 will be returned. 
    Here we assume that the shortest section is a sentence and every sentence must have at least 2 words: subject, verb and object
    :param article_id: int
    :param title: str
    :param text: str
    :param section_by: str
    :return: list [(article_id, text), (article_id, text)]
    """
    assert isinstance(article_id, int)
    assert isinstance(title, str)
    assert isinstance(text, str)
    assert section_by in ("paragraph", "sentence")
    if input_types==["title"]:
        sections = [preprocess_text(title)]
    elif input_types==["text"]:
        if section_by == "paragraph":
            sections = text_to_paragraphs(text)
        elif section_by == "sentence":
            sections = text_to_sentences(text)
        elif section_by is None:
            sections = [text]
        else:
            raise ValueError("Invalid section_by value. Allowed values are None, 'paragraph' and 'sentence'")
        sections = list(filter(lambda x: len(x.split())>2, map(preprocess_text, sections)))
    elif input_types==["title", "text"]:
        if section_by == "paragraph":
            sections = text_to_paragraphs(text)
        elif section_by == "sentence":
            sections = text_to_sentences(text)
        elif section_by is None:
            sections = [text]
        else:
            raise ValueError("Invalid section_by value. Allowed values are None, 'paragraph' and 'sentence'")
        sections = list(filter(lambda x: len(x.split())>2, [preprocess_text(title)] + list(map(preprocess_text, sections))))
    else:
        raise ValueError("Invalid input_types value. Allowed values are ['title'], ['text'] and ['title', 'text']")
    sections_count = len(sections)
    return list(zip([article_id]*sections_count, sections))
    
def preprocess_data(input_filename:str, sample_size=None, section_by:str=None, input_types:list=[])->str:
    """
    Preprocess data
    :param input_filename: str
    :return: dict
    """
    
    logger.info("Preprocessing data ...")
    assert isinstance(input_filename, str)
    assert os.path.exists(input_filename)
    assert isinstance(sample_size, int) or sample_size is None
    assert section_by in ("paragraph", "sentence") or section_by is None

    processed_filename = os.path.join(os.path.dirname(os.path.dirname(input_filename)), "processed", f"{section_by}_{'_'.join(input_types)}_processed.csv")
    article_ids_filename = os.path.join("models", f"{section_by}_{'_'.join(input_types)}_ids.json")
    stats_filename = os.path.join("models", f"{section_by}_{'_'.join(input_types)}_stats.json")
    data = pd.read_csv(input_filename)
    data = data[(~data['title'].isnull()) & (~data['text'].isnull())]
    if sample_size is not None:
        data = data.sample(sample_size)
    input_size = data.shape[0]
    if section_by==None:
        if input_types==["title"]:
            data = list(map(lambda x: (x[0], preprocess_text(x[1])), tqdm.tqdm(data[["article_id", "title"]].values)))
            data = pd.DataFrame(data, columns=["article_id", "text"])
        elif input_types==["text"]:
            data = list(map(lambda x: (x[0], preprocess_text(x[1])), tqdm.tqdm(data[["article_id", "text"]].values)))
            data = pd.DataFrame(data, columns=["article_id", "text"])
        else:
            data = pd.DataFrame(
                list(reduce(lambda x, y: x+y, map(lambda x: [(x[0], preprocess_text(x[1])), (x[0], preprocess_text(x[2]))], tqdm.tqdm(data[["article_id", "title", "text"]].values)))),
                columns=["article_id", "text"]
            )
    elif section_by=="paragraph" or section_by=="sentence":
        data = reduce(lambda x,y: x+y, map(lambda x:preprocess_article(x[0], x[1], x[2], section_by, input_types), tqdm.tqdm(data[["article_id", "title", "text"]].values)))
        data = pd.DataFrame(data, columns=["article_id", "text"])
    else:
        raise ValueError("Invalid section_by value. Allowed values are 'paragraph' and 'sentence' and None")
    
    data["section_id"] = range(data.shape[0])

    data, article_id_to_article_ids, section_id_to_article_id = dedup_data(data)
    data.to_csv(processed_filename, index=False)
    logger.info(f"data size comparison: \n\toriginal size: {input_size} \n\tprocessed size: {data.shape[0]}")

    ids_mapper = {
        "article_id_to_article_ids": article_id_to_article_ids,
        "section_id_to_article_id": section_id_to_article_id,
        "index_id_to_raw_id": None
    }
    utils.save_json(ids_mapper, article_ids_filename)

    # collect stats about sections per each article
    stats = pd.DataFrame(section_id_to_article_id.items(), columns=["section_id", "article_id"])\
        .groupby(["article_id"]).count().describe()\
            .rename(index={"50%":"median"}, columns={"section_id": "sections_by_article"})\
                .loc[["median", "mean", "std"]]\
                    .astype(int).to_dict()
    utils.save_json(stats, stats_filename)
    return {
        "path_to_processed_text": processed_filename,
        "path_to_article_ids": article_ids_filename
    }
    

if __name__ == "__main__":
    # # output_filename = utils.get_raw_data_from_aws_mongo()
    # output_filename = "data/raw/raw.csv"
    # print(preprocess_data(output_filename, sample_size=50, section_by=config.TEXT_SECTION_TYPE, input_types=config.TRAIN_DATA_INPUT_TYPES))
    text = "EY achieves highest growth in nearly two decades, reports record global revenue of US$45.4b"
    r = preprocess_text(text)
    print(r)