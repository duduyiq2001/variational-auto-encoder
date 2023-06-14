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

figsize = 15
fig = plt.figure(figsize=(figsize, 10))
for i in range(16):
    ax = fig.add_subplot(4, 4, i+1)
    ax.axis('off')
    ax.text(0.5, -0.15, str(yLabelValues[yTest[i]]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(xTest[i], cmap = 'gray')
plt.show()
  
