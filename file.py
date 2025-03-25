import pandas as pd
import numpy as np
from dense import Dense
from sigmoid import Sigmoid
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
def predict(input,network):
    err = 0
    for i in range(len(input)):
        [input_feature,expected_output] = polish_input(input,i)
        
        out = forward_prop(network,input_feature)
        err += abs(expected_output-round(out[0][0]))
    return (1-err/len(input))

def polish_input(input_row,index):
        #input = features (first column)
        input = input_row.iat[index,0]
        #expected_output expected result (second column)
        expected_output = input_row.iat[index,1]
        #polish input features (removing "["","]")
        input=input.replace("[","")
        input=input.replace("]","")
        #splitting features in array
        inputs = input.split(", ")
        #converting into np array
        features = np.array(inputs,dtype=float)
        #reshape for transosing correctly
        features = np.reshape(features,(1280,1))
        return (features,expected_output)
def forward_prop(network,input):
    #forward propagation
    out = input
    for layer in network:
        out= layer.forward(out)
    return out
def back_prop(network,gradient,learning_rate):
    for layer in reversed(network):
        gradient = layer.backward(gradient,learning_rate)
        
#legge il file "feke.csv" e lo mescola
fake = pd.read_csv("fake.csv",index_col=0).sample(frac=1)
#legge il file "reali.csv" e lo mescola
true = pd.read_csv("reali.csv",index_col=0).sample(frac=1)

#4622 fake images
#  75% training
#  10% validation
#  15% testing
traing_fake_len = round(len(fake.index)*0.75)
validation_fake_len = round(len(fake.index)*0.10)
testing_fake_len = round(len(fake.index)*0.15)
#2553 true images
# 75% training
# 10% validation
# 15% testing
traing_true_len = round(len(true.index)*0.75)
validation_true_len = round(len(true.index)*0.10)
testing_true_len = round(len(true.index)*0.15)
#creo training set (75% (fake + true))
training = pd.concat([fake[0:traing_fake_len],true[0:traing_true_len]])
#creo validation set (10% (fake + true))
validation = pd.concat([fake[traing_fake_len: traing_fake_len + validation_fake_len ],true[traing_true_len:traing_true_len+validation_true_len]])
#creo testing set (15% (fake + true))
testing = pd.concat([fake[traing_fake_len + validation_fake_len:],true[traing_true_len+validation_true_len:]])

#mescolo training set
training = training.sample(frac=1)
#mescolo validation set
validation = validation.sample(frac=1)
#mescolo testing set
testing = testing.sample(frac=1)



#network
network = [
    Dense(1280,30),
    Sigmoid(),
    Dense(30,30),
    Sigmoid(),
    Dense(30,15),
    Sigmoid(),
    Dense(15,1),
    Sigmoid()
]


#training 
errors_training= []
errors_validation= []
epoches = 600
for k in range(epoches):
    err = 0
    for i in range(len(training)):
        [input,expected_output] = polish_input(training,i)
        
        out = forward_prop(network,input)
        # calulating error mean sqare error
        err += mse(expected_output,out)
        #calculating gradient in output
        grad = mse_prime(expected_output,out)
        #back propagation
        back_prop(network,grad,0.14)
        
    # calculating validation set error every 50 epoches
    if k%10 == 0 :
        err2 = 0
        for j in range(len(validation)):
                [input,expected_output2] = polish_input(validation,j)

                out2 = forward_prop(network,input)
                
                err2 += mse(expected_output2,out2[0][0])
        errors_validation.append(err2/len(validation)*100)
        errors_training.append(err/len(training)*100)
        print(k,err2/len(validation))
        print(err/len(training))

print(predict(testing,network))
plt.plot(errors_validation)
plt.plot(errors_training)
plt.ylabel(' error')
plt.show()





