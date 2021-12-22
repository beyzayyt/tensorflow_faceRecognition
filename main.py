# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Import uuid library to genrate unique image names (universally unique identitfy)
import uuid

# Import metric calculations
from tensorflow.keras.metrics import  Precision, Recall

# Model(inputs=[inputimage, verificationimage], outputs=[1,0])

# class L1Dist(Layer)

# Avoid OOM (out of memory) errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Make the directories
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)

# http://vis-www.cs.umass.edu/lfw/ (dataset)

# Move LFW Images to the following repository data/negative
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', directory)):
#         EX_PATH = os.path.join('lfw', directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)

# Access to WebCam
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     frame = frame[120:120 + 250, 200:200 + 250, :]  # 250x250
#
#     # Collect anchors
#     if cv2.waitKey(1) & 0XFF == ord('a'):
#         # Create uniqe file path
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # Write out anchor image
#         cv2.imwrite(imgname, frame)
#
#     # Collect positives
#     if cv2.waitKey(1) & 0XFF == ord('p'):
#         # Create uniqe file path
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # Write out positive image
#         cv2.imwrite(imgname, frame)
#
#     # Show image back to screen
#     cv2.imshow('Image Collection', frame)
#
#     if cv2.waitKey(1) & 0XFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(300)

dir_test = anchor.as_numpy_iterator()


# print(dir_test.next())


# Preprocessing- Scale and Resize
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be a between 0 and 1
    img = img / 255.0
    return img


# img = preprocess('data\\anchor\\c580542f-51e2-11ec-8482-a44cc883f220.jpg')
# print(img.numpy().min())
# print(img.numpy().max())

# dataset.map(preprocess)

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)
print('LINE 100 data ' + str(data))

samples = data.as_numpy_iterator()
example = samples.next()


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


res = preprocess_twin(*example)  # This star is effectively unpacking the values we have got inside of up tuple
print(len(res))  # 3

# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
# samples = data.as_numpy_iterator()
# samp = samples.next()
# print(samp[2])

# Training partition
train_data = data.take(round(len(data) * .7))
train_data = train_data.batch(16)  # 16 images inside
train_data = train_data.prefetch(8)

# train_samples = train_data.as_numpy_iterator()
# train_sample = train_samples.next()

# Testing partition (FOR TEST)
test_data = data.skip(round(len(data) * .7))
test_data = test_data.take(round(len(data) * .3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Build embedding layer
inp = Input(shape=(100, 100, 3), name='input_image')


# c1 = Conv2D(64,(10,10), activation='relu')(inp)
# m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
#
# # Second block
# c2 = Conv2D(128, (7, 7), activation='relu')(m1)
# m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)
# print(c2)

def make_embedding():
    # First block
    inp = Input(shape=(100, 100, 3), name='input_image')
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()


# Build distance layer
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# Make Siamese Model
def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    Classifier = Dense(1, activation='sigmoid')(distances)
    print(Model(inputs=[input_image, validation_image], outputs=Classifier, name='SiameseNetwork').summary())

    return Model(inputs=[input_image, validation_image], outputs=Classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()
print(siamese_model.summary())

# *** TRAINING ***

# Setup loss and optimizer

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001

# Establish checkpoints
checkpoint_dir = '_/training_checkpoints'  # Defines checkpoint directory
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')  #
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


@tf.function
def train_step(batch):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    #  Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss


# Build training loop
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch)
            progbar.update(idx + 1)

    # Save checkpoints
    if epoch % 10 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

EPOCHS = 50
# train(train_data, EPOCHS)

# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
test_var = test_data.as_numpy_iterator().next()
# print("test var " + str(test_var))
# print(len(test_var[1]))

# Make predictions
y_hat = predictions = siamese_model.predict([test_input,test_val])

# Post processing the results
a = [1 if prediction > 0.5 else 0 for prediction in y_hat]
print(a) # [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
print(y_true)
# res = []
# for prediction in y_hat:
#     if prediction > 0.5:
#         res.append(1)
#     else:
#         res.append(0)

# Creating am metric object
# m = Recall()
m = Recall()
# Calculating the recall value
m.update_state(y_true,y_hat)
# Return Recall Result
m.result().numpy()
print(m.result().numpy())

# Set a plot size
plt.figure(figsize=(18,8))
# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[3])
# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[3])
plt.show()
# Render clearly
print(plt.show())

# Save Model

# Save weights
siamese_model.save('models')

#Reload model
model = tf.keras.models.load_model('models',
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
# siamese_model.load_weights()

# Make predictions with reloaded model
model.predict([test_input,test_val])
# View model summary
model.summary()

# Real time test

# Verification function
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data','verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # Make predictions
        result= model.predict(list(np.expand_dims([input_img,validation_img], axis=1)))
        results.append(result)

    # Detection threshold :  Metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification threshold : Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return  results, verified

# OpenCV Real time verification

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]  # 250x250
    cv2.imshow('Verification', frame)

    # Verification trigger
    if cv2.waitKey(1) & 0XFF == ord('v'):
        # Save input image to application_data/input_image_folder
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(model,0.5, 0.8)
        print(verified)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converter_model.tflite","wb").write(tflite_model)





