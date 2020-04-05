# e4040-2019Fall-Project
This is a implementation of weight normalization in ConvPool-CNN-C

#Prequisite

tensorflow 1.13.1
numpy 1.16.4
python 3.7.3

#Files

ConvPool_CNN_C.py
we define the neural network model in this py file, which includes
first add noise
three convolutionlayer 3*3*96
2*2 maxpool and dropout
three 3*3*192 convolution
2*2 maxpool and dropout
3*3*192, two 1*1*192
global average and dense

WN_Layers.py
we define layers that can be used to build neural netowrks
Batch normalization
Mean-only batch normalization
Three activtion function relu,leaky relu,sofmax
convolution 
dense
add noise
global average pooling
dropout

TrainModel.py,TrainModel.ipynb
they have the same code in different file, if you want to train your own model run it
fisrt set the path to store your result
there are different configs of our network
bn, mobn, wn, wnmbn, n represent batch norm, mena-only batch norm, weight norm, weight norm+ mean-only batch norm, normal
ranodm seed is 12345
epochs is 200
batch size is 100
the dataset is from tensorflow.keras.datasets
the model will be firt initialized then trained, finally tested
the train loss, test loss and train time will be saved to npy file


in resut folder, ther are five sub folders, each sub folder contains its train,test loss, train time, model files, and loss plot









