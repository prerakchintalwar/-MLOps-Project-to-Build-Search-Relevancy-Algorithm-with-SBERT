from flask import Flask, request, jsonify
import datetime
import os
from src import utils
from src import config
from src import build_index
from src import search

app = Flask(__name__)

index = build_index.init_index(index_type=config.SEARCH_INDEX_TYPE)
index.load(os.path.join(config.MODELS_DIR, config.SEARCH_INDEX))
ids_mapper = utils.load_json(os.path.join(config.MODELS_DIR, f"{config.TEXT_SECTION_TYPE}_{'_'.join(config.TRAIN_DATA_INPUT_TYPES)}_ids.json"))
sections_stats = utils.load_json(os.path.join(config.MODELS_DIR, f"{config.TEXT_SECTION_TYPE}_{'_'.join(config.TRAIN_DATA_INPUT_TYPES)}_stats.json"))["sections_by_article"]

@app.route('/ping', methods=['GET'])
def ping():
    """
    Ping the server for heartbeat
    """
    return jsonify({"status": "ok", "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

@app.route('/search', methods=['POST'])
def search_relevant_articles():
    """
    Search for relevant articles
    :param query: list, query to search for
    :param k: int, number of articles to return
    """
    req = request.get_json()
    query = req["query"]
    k = int(req["k"])
    results = search.search(index, query, k, ids_mapper, sections_stats)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)