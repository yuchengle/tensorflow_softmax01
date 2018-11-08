# coding=utf-8
from flask import Flask
import tensorflow_softmax_restore as tfsr

app = Flask(__name__)
PORT=8200

@app.route('/')
def index2():
    file = 'test0.txt'
    return tfsr.getresult(file)

if __name__ == '__main__':
    app.run(debug=True, port=PORT)
 