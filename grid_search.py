from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from keras.models import Sequential
from keras import optimizers

class ModelSearch:
    def __init__(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y

    def create_model(self, learning_rate=0.001, num_filters=16, filter_size=3, lstm_units=32,
                     dense_units=[16, 8]):
        #Define model architecture
        model = Sequential()
        model.add(Conv1D(num_filters, filter_size, activation='relu', padding='same', input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(num_filters*2, filter_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(LSTM(lstm_units))
        for units in dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.2))
        model.add(Dense(3))

        #Compile model
        optimizer = optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def grid_search(self, param_grid, cv=3, batch_size=32):
        model = KerasRegressor(build_fn=self.create_model, verbose=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        grid_result = grid.fit(self.train_X, self.train_Y, batch_size=batch_size)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        mean_test_scores = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']
        for mean, param in zip(mean_test_scores, params):
            print("Test score: %f with: %r" % (mean, param))
