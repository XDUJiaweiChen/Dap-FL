import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from tensorflow.keras.utils import to_categorical
from art.attacks.inference import model_inversion 
from art.estimators.classification import KerasClassifier

model_path=""

def load_mnist():
  """Loads MNIST-Dataset and preprocesses to combine training and test data."""
  
  # load the existing MNIST digit dataset that comes in form of traing + test data and labels
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  # scale the images from color values 0-255 to numbers from 0-1 to help the training process
  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  # convolutional layers expect images to have 3 dimensions (width, height, depth)
  # in color images the depth is 3 for the RGB channels
  # MNIST is grayscale and hence originally does not need a third dimension
  # so we need to artificially add it
  train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
  test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
  return train_data, train_labels, test_data, test_labels

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

train_data, train_labels, test_data, test_labels = load_mnist()

# load model
model = tf.keras.models.load_model(model_path)

# convert keras model to ART model
classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

# train the model
history = classifier.fit(train_data, to_categorical(train_labels),
           batch_size=256, nb_epochs=1)


# create the attack object
my_attack = model_inversion.MIFace(classifier)

# create an array of the classes to be attacked
y_all = np.arange(10)

# inversion model 
inferred_images = my_attack.infer(x=None,y=y_all,max_iter=10000, learning_rate=0.1)

# plot the inverted class representations
num_row = 2
num_col = 5
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(10):
    ax = axes[i//num_col, i%num_col]
    ax.set_axis_off()
    ax.imshow(inferred_images[i,:].reshape(28,28), cmap='gray')
    ax.set_title('Label: {}'.format(y_all[i]))
plt.tight_layout()
plt.show()
