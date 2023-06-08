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
numLayers = 2
inputShape = (28, 28, 1)
layerInputs = [[32, 3, 1, None],[64, 3, 2, None]]
encoderParams = [inputShape, numLayers, layerInputs, numLatentVars]


# decoder parameters
inputShape = (2,)
denseSize = 14 * 14 * 64
reshapeSize = (14, 14, 64)
numLayers = 1 # should be 1 less than numLayers in encoder
layerInputs = [
    [64, 3, 2, None],
]
outputInputs = [1, 3, 1, 'sigmoid']
decoderParams = [inputShape, denseSize, reshapeSize, numLayers, layerInputs, outputInputs]

batchSize = 128
numLatentVars = 2
epochs = 3
trainLength = 60000 # number of images to use in training, this is half the maximum
learningRate = 0.001
VAE = fashionVAE(encoderParams, decoderParams, batchSize, numLatentVars, epochs, trainLength, learningRate, xTrain, xTest, yTest, xTestReshaped, yLabelValues, seed)

VAE.train()

VAE.generateImages()

VAE.testTrainModel()

VAE.testModel()

"""
Time for training is 311.0560004711151
Average Loss for predicting 10000 training images: 211.3249969482422
Time for predicting 10000 test images is 11.860170602798462 sec, Average loss: 211.3617706298828
"""