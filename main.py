from flask import Flask, render_template, request, jsonify
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers

app = Flask(__name__)

new_document_store = FAISSDocumentStore.load("my_faiss")

retriever = EmbeddingRetriever(
    document_store=new_document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

pipe = ExtractiveQAPipeline(reader, retriever)

from urllib.parse import unquote

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = unquote(request.form['query'])
        print("Decoded Query:", query)
        prediction = pipe.run(
            query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )
        answer = prediction["answers"][0].answer
        
        return f"Answer: {answer}"
    
    return render_template("index.html")



@app.route('/about', methods=['GET'])
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)
