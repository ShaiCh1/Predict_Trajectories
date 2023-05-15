import glob
import random
import numpy as np
import pandas as pd
from Model import LSTMModel, SimpleFeedForwardModel
from data_preprocess import DataProcessor
import os
import matplotlib.pyplot as plt

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
def create_traj_plot(model_name,lstm_model, test_X_Normalize, test_X, test_Y):
    # Select a random object id to examine
    object_ids = np.unique(test_X[:,:,0])
    object_id = random.choice(object_ids)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Find the test sequences for the object
    test_sequence_indices = np.where(test_X[:, :, 0] == object_id)[0]

    # Plot the expected trajectory
    expected_data = []
    expected_output_labels = []
    for i, test_sequence_index in enumerate(test_sequence_indices):
        # Get the expected output (actual trajectory)
        expected_output = test_Y[test_sequence_index]
        expected_output = expected_output.reshape((-1, 3))

        # Extract x, y, and z coordinates
        x_expected = expected_output[:, 0]
        y_expected = expected_output[:, 1]
        z_expected = expected_output[:, 2]
        # Store the predicted values in a dictionary
        row = {'x': float(x_expected), 'y': float(y_expected), 'z': float(z_expected)}

        # Append the dictionary to the list
        expected_data.append(row)

    # Create a dataframe from the list of dictionaries
    df_expected = pd.DataFrame(expected_data)

    # Plot the predicted trajectories
    predicted_data = []
    predicted_output_labels = []
    for i, test_sequence_index in enumerate(test_sequence_indices):
        # Get the input sequence
        input_sequence = test_X_Normalize[test_sequence_index]

        # Use the model to predict the trajectory
        predicted_output = lstm_model.model.predict(np.array([input_sequence]))
        predicted_output = predicted_output.reshape((-1, 3))

        # Extract x, y, and z coordinates
        x_predicted = predicted_output[:, 0]
        y_predicted = predicted_output[:, 1]
        z_predicted = predicted_output[:, 2]
        # Store the predicted values in a dictionary
        row = {'x': float(x_predicted), 'y': float(y_predicted), 'z': float(z_predicted)}

        # Append the dictionary to the list
        predicted_data.append(row)

    # Create a dataframe from the list of dictionaries
    df_predicted = pd.DataFrame(predicted_data)

    df_expected = df_expected.tail(3)
    df_predicted = df_predicted.tail(3)
    print(df_expected)
    print(df_predicted)
    # Plot the trajectories
    ax.plot(df_expected['x'], df_expected['y'], label='Expected')
    ax.plot(df_predicted['x'], df_predicted['y'] ,label='Predicted')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    plt.title(model_name)

    plt.legend()
    plt.show()
    return df_predicted,df_expected

def main():
    np.random.seed(123)
    # File paths
    folder_path = "data"
    # Data preprocessing
    data_processor = DataProcessor(glob.glob(os.path.join(folder_path, '*.txt')))
    train_X, train_Y, test_X_Normalize, test_Y,test_X = data_processor.get_train_test_data()

    # Model training and evaluation
    # best_model = ModelSearch(train_X, train_Y) #hyper tuning
    # best_params = best_model.grid_search(param_grid, cv=3).best_params_  #Get best params
    lstm_model = LSTMModel(train_X, train_Y, test_X_Normalize, test_Y)
    lstm_model.train_model(epochs=16, batch_size=128, learning_rate=0.001, train_X=train_X, train_Y=train_Y,test_X=test_X_Normalize, test_Y=test_Y)
    simple_ff_model = SimpleFeedForwardModel(input_shape=train_X.shape[1:])
    simple_ff_model.train_model(epochs=16, batch_size=128, learning_rate=0.001, train_X=train_X, train_Y=train_Y,
                                test_X=test_X_Normalize, test_Y=test_Y)

    # Make predictions on the test set
    df_predicted_list = []
    df_expected_list = []
    for i in range(5):
        df_predicted, df_expected = create_traj_plot("lstm_model", lstm_model, test_X_Normalize, test_X, test_Y)
        df_predicted_list.append(df_predicted)
        df_expected_list.append(df_expected)
        df_predicted, df_expected = create_traj_plot("simple_ff_model", simple_ff_model, test_X_Normalize, test_X,
                                                     test_Y)
        df_predicted_list.append(df_predicted)
        df_expected_list.append(df_expected)

    # Concatenate all the predicted and expected dataframes into a final dataframe
    df_predicted_all = pd.concat(df_predicted_list, axis=0, ignore_index=True)
    df_expected_all = pd.concat(df_expected_list, axis=0, ignore_index=True)

    # Save the predicted and expected dataframes to CSV files
    df_predicted_all.to_csv("predicted_data.csv", index=False)
    df_expected_all.to_csv("expected_data.csv", index=False)


if __name__ == "__main__":
    main()