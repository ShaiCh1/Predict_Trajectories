import glob
from Model import LSTMModel
from data_preprocess import DataProcessor
import os
from grid_search import ModelSearch


# Define the grid search parameters
param_grid = {
    'batch_size': [32, 64],
    'epochs': [16, 32, 64],
    'learning_rate': [0.001, 0.01],
    'num_filters': [32, 64, 128],
    'filter_size': [3, 5, 7],
    'lstm_units': [64, 128, 256],
    'dense_units': [[32, 16, 8], [64, 32, 16]]
}

if __name__ == '__main__':
    # File paths
    folder_path = "data"

    # Data preprocessing
    data_processor = DataProcessor(glob.glob(os.path.join(folder_path, '*.txt')))
    train_X, train_Y, test_X, test_Y = data_processor.get_train_test_data()


    # Model training and evaluation
    #best_model = ModelSearch(train_X, train_Y)
    #best_params = best_model.grid_search(param_grid, cv=3).best_params_  #Get best params
    lstm_model = LSTMModel(train_X, train_Y, test_X, test_Y)
    lstm_model.train_model(epochs=16, batch_size=128, learning_rate=0.001)  #or **best_params for best model
    lstm_model.save_model('my_model')
