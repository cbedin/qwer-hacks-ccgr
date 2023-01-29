#Load CIFAR10
CIFAR10_dataset = tf.keras.datasets.cifar10.load_data()
 
(train_images, train_labels), (test_images, test_labels) = CIFAR10_dataset

#name the classes 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#format image pixels
train_images = train_images / 255.0
test_images = test_images / 255.0

#train model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

#evaluate model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#probability model

probability_model = tf.keras.Sequential([model,                              tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images) 