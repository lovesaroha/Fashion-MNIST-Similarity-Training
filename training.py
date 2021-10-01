# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model on Fashion MNIST dataset to show similarity between two images.
import numpy
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import backend as K
import random

# Parameters.
learningRate = 0.01
epochs = 15
batchSize = 256

# Load fashion MNIST dataset.
mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (validation_images,
                                     validation_labels) = mnist.load_data()

# Normalize inputs.
training_images = training_images.astype("float32") / 255
validation_images = validation_images.astype("float32") / 255

# Create image pairs.
def create_pairs(x, digit_indices):
  pairs = []
  labels = []
  n = min([len(digit_indices[d]) for d in range(10)]) - 1
  for d in range(10):
    for i in range(n):
      imageOne, imageTwo = digit_indices[d][i] , digit_indices[d][i+1]
      pairs += [[x[imageOne] , x[imageTwo]]]
      inc = random.randrange(1,10)
      dn = (d + inc) % 10
      imageOne, imageTwo = digit_indices[d][i] , digit_indices[dn][i]
      pairs += [[x[imageOne] , x[imageTwo]]]
      labels += [1,0]
  return numpy.array(pairs) , numpy.array(labels)

# Pairs on set (if similar label return 0 else 1).
def create_pairs_on_set(images, labels):
  digit_indices = [numpy.where(labels == i)[0] for i in range(10)]
  pairs, labels = create_pairs(images, digit_indices)
  labels = labels.astype("float32")
  return pairs, labels

# Prepare training and validation data.
training_data, training_output = create_pairs_on_set(training_images, training_labels)
validation_data, validation_output = create_pairs_on_set(validation_images, validation_labels) 

# Calculate euclidean distance.
def euclidean_distance(vectors):
  x , y = vectors
  sum_of_square = K.sum(K.square(x-y) , axis=1, keepdims=True)
  return K.sqrt(K.maximum(sum_of_square ,K.epsilon()))

# Euclidean output shape.
def euclidean_output_shape(shapes):
  shape1, shape2 = shapes
  return (shape1[0] , 1)

# Create a  base model.
def base_model():
  input_layer = keras.layers.Input(shape=(28,28,) , name="base_input")
  x = keras.layers.Flatten()(input_layer)
  x = keras.layers.Dense(units=128 , activation="relu")(x)
  x = keras.layers.Dropout(0.1)(x)
  x = keras.layers.Dense(units=128 , activation="relu")(x)
  x = keras.layers.Dropout(0.2)(x)
  x = keras.layers.Dense(units=128 , activation="relu")(x)
  return keras.models.Model(inputs=input_layer , outputs=x)

base_network = base_model()

# Network one.
input_one = keras.layers.Input(shape=(28,28,) , name="input_one")
output_one = base_network(input_one)

# Network two.
input_two = keras.layers.Input(shape=(28,28,) , name="input_two")
output_two = base_network(input_two)

# Create final output layer. 
output_layer = keras.layers.Lambda(euclidean_distance, name="output_layer" , output_shape=euclidean_output_shape)([output_one , output_two])

# Create a model with two inputs.
model = keras.models.Model(inputs=[input_one , input_two] , outputs=output_layer)

# Custom loss function.
def loss_function(output, prediction):
    square_prediction = K.square(prediction)
    margin_square = K.square(K.maximum(1 - prediction, 0))
    return K.mean(output * square_prediction + (1-output) * margin_square)

# Set loss function and optimizer.
model.compile(loss=loss_function,
              optimizer="adam")

# Train model.
model.fit([training_data[:,0], training_data[:,1]] , training_output , epochs=epochs, batch_size=batchSize , validation_data=([validation_data[:,0], validation_data[:,1]], validation_output))


# Show images.
def show_images(imageOne, imageTwo):
  f = plt.figure()
  f.add_subplot(1, 2, 1)
  plt.imshow(imageOne)
  f.add_subplot(1,2, 2)
  plt.imshow(imageTwo)
  plt.show(block=True)

# Predict on a random pair of images.
index = 7
prediction = model.predict([validation_data[index][0].reshape(1, 28, 28) , validation_data[index][1].reshape(1, 28, 28)])

# Show image pairs.
show_images(validation_data[index][0] , validation_data[index][1])

# Show prediction.
if prediction[0][0] > 0.5:
  print("Not similar")
else:
  print("Similar")