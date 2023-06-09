from vae import *

(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1).astype('float32')/255.0
# xTrain = xTrain[:1500] # first 100 images
# Batch + shuffle data
seed = 60000 # seed for shuffling
xTestReshaped = xTest.reshape(xTest.shape[0], 28, 28, 1).astype('float32')/255.0 # necessary when testing

# Create dictionary of target classes
# Index corresponds to each label
yLabelValues = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# encoder parameters
numLatentVars = 2
numLayers = 4
inputShape = (28, 28, 1)
layerInputs = [[32, 4, 1, None],[64, 4, 2, None],[64, 4, 2, None],[64, 4, 1, None]]
encoderParams = [inputShape, numLayers, layerInputs, numLatentVars]


# decoder parameters
inputShape = (2,)
denseSize = 7 * 7 * 64
reshapeSize = (7, 7, 64)
numLayers = 3 # should be 1 less than numLayers in encoder
layerInputs = [
    [64, 4, 1, None],
     [64, 4, 2, None],
      [32, 4, 2, None]
]
outputInputs = [1, 4, 1, 'sigmoid']
decoderParams = [inputShape, denseSize, reshapeSize, numLayers, layerInputs, outputInputs]

batchSize = 128
numLatentVars = 2
epochs = 3
trainLength = 60000 # number of images to use in training, this is half the maximum
learningRate = 0.0005
VAE = fashionVAE(encoderParams, decoderParams, batchSize, numLatentVars, epochs, trainLength, learningRate, xTrain, xTest, yTest, xTestReshaped, yLabelValues, seed, 'model5')

VAE.train()

VAE.generateImages()

VAE.testTrainModel()

VAE.testModel()

"""
Time for training is 404.5199246406555
Average Loss for predicting 10000 training images: 43.819000244140625
Time for predicting 10000 test images is 12.228641033172607 sec, Average loss: 43.83798599243164
"""