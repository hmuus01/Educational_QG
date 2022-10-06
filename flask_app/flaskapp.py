#This file serves as the client, it is made with flask and receives data from the server which is displayed to the user
#The server and the client communicate via sockets obtained and adapted the code for this from https://realpython.com/python-sockets/

#Import Statements
import os
import random

from flask import Flask, request, redirect, render_template, url_for
import pandas as pd
import socket

# from model_factory import ModelFactory
# from utils import generate

app = Flask(__name__)

#1 The client: reads in a validation(test) dataset
path = os.path.join('..', 'data', 'lectures.csv')
print('reading csv')
data_df = pd.read_csv(path, header=None, sep='\t')


#obtained and adapted from https://realpython.com/python-sockets/
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

#Default route to the homepage
@app.route('/', methods=["GET", "POST"])
def home():
    #render the homepage
    return render_template('home.html')


n_questions = 0
good_questions = 0

@app.route('/questions', methods=["GET", "POST"])
def questions():
    global good_questions
    global n_questions

    i = random.randint(0, data_df.shape[0] - 1)
    context = data_df.iloc[i].values[0]
    if request.method == 'POST':
        option = request.form['option']
        n_questions+=1
        good_questions += option=='good'
        result = good_questions/n_questions
        print(result)
        print(f'num questions asked: {n_questions}')
        print(f'num good questions : {good_questions}')
        print("------------------------------------------")
    input_answer = '[MASK]'
    # model_factory = ModelFactory()
    model = 'leaf'
    # best_model, tokenizer = model_factory.get(model)

    # question, _, __, ___ = generate(best_model, tokenizer, input_answer, context)
    """Landing page."""
    return render_template(
        'questions.html',
        title="Demo CS Question",
        description=context
    )
# Python assignes main to a script when its being executed | if the condition below is satisfied then the app will be executed
if __name__ == '__main__':
    app.run()
