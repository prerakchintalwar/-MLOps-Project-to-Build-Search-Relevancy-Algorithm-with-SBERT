"""
For each of the models created, we will run similarity search on a set of queries and evaluate the results using recall, precision, and F1 score. 
We will also evaluate the time taken to run similarity search on the queries.

Models to be validated:
1. None_title.annoy
    - no sectioning of long texts
    - only title embeddings
    - use of annoy as similarity index
2. None_title.faiss
    - no sectioning of long texts
    - only title embeddings
    - use of faiss as similarity index
3. None_text.annoy
    - no sectioning of long texts
    - only text embeddings
    - use of annoy as similarity index
4. None_text.faiss
    - no sectioning of long texts
    - only text embeddings
    - use of faiss as similarity index
5. None_title_text.annoy
    - no sectioning of long texts
    - title and text embeddings
    - use of annoy as similarity index
6. None_title_text.faiss
    - no sectioning of long texts
    - title and text embeddings
    - use of faiss as similarity index
7. paragraph_title.annoy
    - sectioning of long texts into paragraphs
    - only title embeddings
    - use of annoy as similarity index
8. paragraph_title.faiss
    - sectioning of long texts into paragraphs
    - only title embeddings
    - use of faiss as similarity index
9. paragraph_text.annoy
    - sectioning of long texts into paragraphs
    - only text embeddings
    - use of annoy as similarity index
10. paragraph_text.faiss
    - sectioning of long texts into paragraphs
    - only text embeddings
    - use of faiss as similarity index
11. paragraph_title_text.annoy
    - sectioning of long texts into paragraphs
    - title and text embeddings
    - use of annoy as similarity index
12. paragraph_title_text.faiss
    - sectioning of long texts into paragraphs
    - title and text embeddings
    - use of faiss as similarity index
"""
import pandas as pd
import random
import os
try:
    from src import config
    from src import utils
except:
    import config
    import utils

def sample_sentences(text:str, n:int=1)->list:
    """
    sample N sentences text
    """
    assert isinstance(text, str)
    assert isinstance(n, int)
    assert n>0
    sentences = text.split(".")
    if len(sentences)<n:
        return sentences
    else:
        return random.sample(sentences, n)

def sample_validate_set(n:int=100, k:int=1)->list:
    """
    sample a validation set of N queries from the test set
    """
    assert isinstance(n, int)
    assert n>0
    df = pd.read_csv(os.path.join(config.DATA_DIR, "raw", "raw.csv"))[["article_id", "title", "text"]].dropna().sample(n)
    df['sentence'] = df.apply(lambda x: ". ".join(sample_sentences(x["text"], n=k)), axis=1).str.replace("\n", " ")
    data = df[["article_id", "title", "sentence"]].to_dict(orient="records")
    return data

def build_gpt_prompt(data:list, prompt_type=None, save=False)->str:
    """
    build gpt prompt from a dataframe
    """
    assert isinstance(data, list)
    assert prompt_type in ("question", "restyle")
    
    if prompt_type=="question":

        prompt = f"""
        For each of the following {len(data)} pairs of {{title || sentence}}, generate 1 single question that can be answered by the sentence or the title:
        """
        result = []
        for i, d in enumerate(data):
            if len(d["sentence"].split(" "))<3:
                continue
            prompt += f"""\n
            {i+1}. {{{d["title"]} || {d["sentence"]}}}
            """
            result.append(d)
        if save:
            utils.save_json({"prompt": prompt, "data": result}, os.path.join(config.DATA_DIR, "raw", "gpt_prompt.json"))
        return prompt
    if prompt_type=="restyle":

        prompt = f"""
        Rewrite each of the following {len(data)} sentence into a new style of your choosing:
        """
        result = []
        for i, d in enumerate(data):
            if len(d["sentence"].split(" "))<3:
                continue
            prompt += f"""\n
            {i+1}. {d["sentence"]}
            """
            result.append(d)
        if save:
            utils.save_json({"prompt": prompt, "data": result}, os.path.join(config.DATA_DIR, "raw", "gpt_prompt.json"))
        return prompt
    raise(Exception("prompt_type must be either 'question' or 'restyle'"))

if __name__=="__main__":
    data = sample_validate_set(n=50)
    prompt = build_gpt_prompt(data, prompt_type="restyle", save=True)
    print(len(prompt.split(" ")))