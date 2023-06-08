import os
import time
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np

(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1).astype('float32')/255.0
# xTrain = xTrain[:1500] # first 100 images
# Batch + shuffle data
seed = 60000 # seed for shuffling
xTestReshaped = xTest.reshape(xTest.shape[0], 28, 28, 1).astype('float32')/255.0 # necessary when testing

# Create dictionary of target classes
# Index corresponds to each label
yLabelValues = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

class fashionVAE():
  def __init__(self, encoderParams, decoderParams, batchSize, numLatentVars, epochs, trainLength, learningRate):
    self.epochs = epochs
    self.enc = self.encoder(*encoderParams)
    self.enc.output
    self.dec = self.decoder(*decoderParams)
    self.dec.output
    self.lossTrain = []
    self.lossTest = []
    self.timeTrain = []
    self.timeTest = []
    self.times = []
    self.samplingLayer = self.sampling((numLatentVars,), (numLatentVars,))
    self.optimizer = tf.keras.optimizers.legacy.Adam(lr = learningRate)
    self.trainBatch = tf.data.Dataset.from_tensor_slices(xTrain[:trainLength]).shuffle(seed).batch(batchSize)

  def sampling(self, input1,input2):
    def samplingModelLambda(inputParams):
      mean, log_var = inputParams
      epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
      # print(epsilon)
      return mean + K.exp(log_var / 2) * epsilon
    samplingModelLambda
    mean = keras.Input(shape=input1, name='input_layer1')
    log_var = keras.Input(shape=input2, name='input_layer2')
    out = layers.Lambda(samplingModelLambda, name='encoder_output')([mean, log_var])
    enc_2 = tf.keras.Model([mean,log_var], out,  name="Encoder_2")
    return enc_2 

  def encoder(self, inputShape, numLayers, layerInputs, numLatentVars):
    # print(inputShape, numLayers, layerInputs, numLatentVars)
    inputs = keras.Input(shape=inputShape, name='input_layer')
    for i in range(numLayers):
      filters, kernel_size, strides, activation = layerInputs[i]
      if i == 0:
          curLayer = layers.Conv2D(filters, kernel_size=kernel_size, strides= strides, activation=activation, padding='same', name=f'conv2D{i+1}')(inputs)
          curLayer = layers.BatchNormalization(name=f'bn{i+1}')(curLayer)
          curLayer = layers.LeakyReLU(name=f'lReLU{i+1}')(curLayer)
      else:
          curLayer = layers.Conv2D(filters, kernel_size=kernel_size, strides= strides, activation=activation, padding='same', name=f'conv2D{i+1}')(curLayer)
          curLayer = layers.BatchNormalization(name=f'bn{i+1}')(curLayer)
          curLayer = layers.LeakyReLU(name=f'lReLU{i+1}')(curLayer)

    curLayer = layers.Flatten()(curLayer)
    mean = layers.Dense(numLatentVars, name='mean')(curLayer)
    log_var = layers.Dense(numLatentVars, name='logVar')(curLayer)
    model = tf.keras.Model(inputs, (mean, log_var), name="encoder")
    model.summary()
    return model

  def decoder(self, inputShape, denseSize, reshapeSize, numLayers, layerInputs, outputInputs):
    # print(inputShape, denseSize, reshapeSize, numLayers, layerInputs, outputInputs)
    inputs = keras.Input(shape=inputShape, name='inputLayer')
    curLayer = layers.Dense(denseSize, name='dense1')(inputs)
    curLayer = layers.Reshape(reshapeSize, name='reshapeLayer')(curLayer)
  
    for i in range(numLayers):
          filters, kernel_size, strides, activation = layerInputs[i]
          curLayer = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides= strides, activation=activation, padding='same', name=f'convTranspose2D{i+1}')(curLayer)
          curLayer = layers.BatchNormalization(name=f'bn{i+1}')(curLayer)
          curLayer = layers.LeakyReLU(name=f'lReLU{i+1}')(curLayer)

    filters, kernel_size, strides, activation = outputInputs
    outputs = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same', name=f'convTranspose2D{numLayers+1}')(curLayer)
    model = tf.keras.Model(inputs, outputs, name="decoder")
    model.summary()
    return model

  def mse_loss(self, xTrue, xPred):
    r_loss = K.mean(K.square(xTrue - xPred), axis = [1,2,3])
    return 1000 * r_loss

  def get_kl_loss(self, mean, log_var):
      kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
      return kl_loss

  def vae_loss(self, xTrue, xPred, mean, log_var):
      r_loss = self.mse_loss(xTrue, xPred)
      kl_loss = self.get_kl_loss(mean, log_var)
      return  r_loss + kl_loss

  # Notice the use of `tf.function`
  # This annotation causes the function to be "compiled".
  @tf.function
  def train_step(self, images):
      with tf.GradientTape() as encoderTape, tf.GradientTape() as decoderTape:
          mean, log_var = self.enc(images, training=True)
          latent = self.samplingLayer([mean, log_var])
          generated_images = self.dec(latent, training=True)
          loss = self.vae_loss(images, generated_images, mean, log_var)
          self.lossTrain.append(loss)

      encoderGradients = encoderTape.gradient(loss, self.enc.trainable_variables)
      decoderGradients = decoderTape.gradient(loss, self.dec.trainable_variables)
      
      self.optimizer.apply_gradients(zip(encoderGradients, self.enc.trainable_variables))
      self.optimizer.apply_gradients(zip(decoderGradients, self.dec.trainable_variables))
      return loss

  def train(self):
    for epoch in range(self.epochs):
        start = time.time()
        i = 0
        loss_ = []
        for image_batch in self.trainBatch:
            i += 1
            loss = self.train_step(image_batch)
            # print(f'loss len gth: {len(loss)}')
            if i % 5 == 0:
              print(f'Image batch {i}, loss: {tf.reduce_mean(loss)}')
            loss_.append(tf.reduce_mean(loss))
        totTime =  time.time()-start
        self.timeTrain.append(totTime)
        print('Time for epoch {} is {} sec'.format(epoch, totTime))
    print(f'Time for all epochs is {sum(self.timeTrain)}')

  def generateImages(self):
    figsize = 15
    m, v = self.enc.predict(xTest[:16]/255.0)
    latent = self.samplingLayer([m,v])
    reconst = self.dec.predict(latent)
    fig = plt.figure(figsize=(figsize, 10))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i+1)
        ax.axis('off')
        ax.text(0.5, -0.15, str(yLabelValues[yTest[i]]), fontsize=10, ha='center', transform=ax.transAxes)
        ax.imshow(reconst[i, :,:,0]*255, cmap = 'gray')
    plt.show()

  def testTrainModel(self):
    start = time.time()
    mean, log_var = self.enc.predict(xTrain)
    latent = self.samplingLayer([mean, log_var])
    generated_images = self.dec.predict(latent)
    loss = self.vae_loss(xTrain, generated_images, mean, log_var)
    # self.lossTest.append(tf.reduce_mean(loss))
    totTime =  time.time()-start
    # self.timeTest.append(totTime)
    # divide totTime by six to make time comparable to
    print ('Time for predicting 10000 training images is {} sec, loss: {}'.format(totTime/6, tf.reduce_mean(loss)))


  def testModel(self):
    start = time.time()
    mean, log_var = self.enc.predict(xTestReshaped)
    latent = self.samplingLayer([mean, log_var])
    generated_images = self.dec.predict(latent)
    loss = self.vae_loss(xTestReshaped, generated_images, mean, log_var)
    self.lossTest.append(tf.reduce_mean(loss))
    totTime =  time.time()-start
    self.timeTest.append(totTime)
    print ('Time for predicting 10000 test images is {} sec, loss: {}'.format(totTime, sum(self.lossTest)/len(self.lossTest)))
  