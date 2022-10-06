#This File contains the code for the server which uses the three models in this project to make the class
# prediction and send back the result to the client for it to be displayed.
#!/usr/bin/env python3
# Import Statements
import socket
import pickle
from os import path
import numpy as np

# PLEASE NOTE THE CODE FOR THE SERVER HAS BEEN OBTAINED AND ADAPTED FROM THE FOLLOWING LINK https://realpython.com/python-sockets/


# --------------------------------#
# STORE THE THREE MODELS USED IN THIS SYSTEM
models = ['rf', 'svm', 'lr']
#--------------------------------#
#Probability threshold to classify a transaction
proba_threshold = 0.5

#Method to that takes one of the models as a parameter and returns the mod
def get_model(model_str):
    #get the model passed as a parameter (user selected) from the folder 'models'
    file = open(path.join('models', model_str+'.pkl'), 'rb')
    # load the saved machine learning model object(Model and Features)
    clf_obj = pickle.load(file)
    #Print the above machine learning model object | for debugging
    print("clf_obj = " + str(clf_obj))
    # Store the ML model | first element in the object
    model = clf_obj[0]
    # Print the model
    print("model = " + str(model))
    # Store the features found to be the best for the given model | second element in the object
    features_saved = clf_obj[1]
    # Print the features
    print("Mask = " +str(features_saved))
    # return the selected model and the best features for that model
    return model, features_saved

# Funtion that takes the user selected transaction row and makes a prediction using the selected model
def infer(row):
    #response_tokens=[]
    #for x in row.split():
    #    response_tokens.append(x)
    # row = 'x1 x2 x3'
    # row.split['x1', 'x2', 'x3']
    #split the list string sent from the client into a list of strings and store them
    response_tokens = [x for x in row.split()]
    #print string sent from the client split up
    print("response_tokens = " + str(response_tokens))
    model_str = response_tokens[0]
    #Load the model (clf) selected by the user and its saved features
    clf, saved_features = get_model(models[int(model_str)])

    #store the features starting from the second element, excluding the first element(model)
    features_all = [float(x) for x in response_tokens[1:]]
    print("features_all = " + str(features_all))
    #From all the features select only the best features from the selectkbest test
    features = [x for idx, x in enumerate(features_all) if saved_features[idx]]
    #Print the features
    print("Below is Features//")
    print(features)
    print('//')
    #Print the boolean from selectkbest get_support method
    print("Below is Mask#")
    print(saved_features)
    print('#')

    # acquire the features in a numpy array data structure
    X_test = np.array([np.array(features)])

    # pass the features array to the model to predict the probability of the label
    probs = clf.predict_proba(X_test)
    #store just the fraudulent class probabilities
    fraudulent_class_probabilities = probs[:, 1]
    #Print both of the above
    print("prob for both "+str(probs))
    print("Prob its a 1 " +str(fraudulent_class_probabilities))
    #predicts 1 if it greater than the threshold or 0 if not
    y_pred = [1 if x >= proba_threshold else 0 for x in fraudulent_class_probabilities]
    print("y_pred " + str(y_pred))
    #Store the prediction and probabilty transaction is fraud
    response =str(y_pred[0]) + '_'+ str("{:.0f}".format(fraudulent_class_probabilities[0] * 100))
    #Return the result (prediction + probability)
    return response

# obtained and adapted from https://realpython.com/python-sockets/
HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

#The following lines of code are the server code obtained from https://realpython.com/python-sockets/

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                # 5 The server receives the transaction features and the model
                data = conn.recv(1024)
                if not data:
                    break
                 # Print what the server receives
                print("What the server receives")
                print('-----------')
                print(data.decode('utf-8'))
                print("data decode = " + str(data.decode('utf-8')))
                print('-----------')
                # 6 The server uses the chosen model to infer using the features as input
                response = infer(data.decode('utf-8'))
                print("Response from server")
                print(response)
                # 7 The server sends the result ( 0 or 1 and probability) back to the client
                conn.sendall(response.encode('utf-8'))