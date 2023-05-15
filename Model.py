import keras
from keras import Sequential
from keras.layers import Flatten, Dense, LSTM
from matplotlib import pyplot
from sklearn.metrics import r2_score

class BaseModel:
    def __init__(self):
        self.model = None

    def build_model(self):
        pass

    def train_model(self, epochs, batch_size, learning_rate, train_X, train_Y, test_X, test_Y):
        if not self.model:
            self.build_model()
        # Compile model
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate))

        # Train model
        history = self.model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size,
                                 validation_data=(test_X, test_Y))

        # Evaluate model on test set
        test_loss = self.model.evaluate(test_X, test_Y)
        print("Test MSE:", test_loss)

        test_predictions = self.model.predict(test_X)
        test_r2 = r2_score(test_Y, test_predictions)
        print("Test R-squared:", test_r2)

        # Plot loss during training
        pyplot.title('Loss / Mean Squared Error')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    def save_model(self, filename):
        self.model.save(filename)


class LSTMModel(BaseModel):
    def __init__(self, train_X, train_Y, test_X, test_Y):
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

    def build_model(self):
        # Define model architecture
        model = Sequential()
        model.add(LSTM(256, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.train_Y.shape[1]))
        self.model = model


class SimpleFeedForwardModel(BaseModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(9,10)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(3))
        self.model.compile(loss='mse', optimizer='adam')
