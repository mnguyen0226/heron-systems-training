import tensorflow as tf
import datetime

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalized the dataset

def create_model():
    """Create a model"""
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activateion = 'softmax')
    ])

def model_fit():
    model = create_model()
    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    model_fit()