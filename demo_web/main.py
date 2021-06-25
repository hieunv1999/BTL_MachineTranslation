import sys
import torch
from flask import Flask, jsonify, render_template, request
import time
sys.path.append('../MachineTrainslation')
from MachineTrainslation.evaluate import predict, create_model_for_inference

checkpoint_file = '../nmt-model-lstm-90.pth'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
source_vocab = checkpoint['source']
target_vocab = checkpoint['target']
model = create_model_for_inference(source_vocab, target_vocab)
model.load_state_dict(checkpoint['model_state_dict'])

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('main.html', reload = time.time())

@app.route("/api/calc")
def add():
    text = request.args.get('a', 0)
    result = predict(model, source_vocab, target_vocab,text)
    return jsonify({
        "result"        :  result
    })

# @app.route('/', methods=['POST'])
# def my_form_post():
#     text = request.form['text']



if __name__ == '__main__':
    app.run()
