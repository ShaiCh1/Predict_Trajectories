import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, file_names):
        self.file_names = file_names

    def extract_features(self, file_path):
        features = []
        # Open the file for reading
        with open(file_path, 'r') as f:
            # Extract the features
            for line in f:
                fields = line.strip().split()
                frame_id = int(fields[0])
                object_id = int(fields[1])
                object_type = int(fields[2])
                position_x = float(fields[3])
                position_y = float(fields[4])
                position_z = float(fields[5])
                object_length = float(fields[6])
                object_width = float(fields[7])
                object_height = float(fields[8])
                heading = float(fields[9])
                features.append([frame_id, object_id, object_type, position_x, position_y, position_z,
                                  object_length, object_width, object_height, heading])
        return np.array(features)

    def get_file_info(self):
        all_features = []
        for file_name in self.file_names:
            features = self.extract_features(file_name)
            all_features.append(features)
        return all_features
    def get_train_test_data(self, seq_length=10, test_size=0.2, random_state=42):
        all_features = self.get_file_info()
        # Group data by object_id
        data_by_object = {}
        for features in all_features:
            for item in features:
                object_id = item[1]
                if object_id not in data_by_object:
                    data_by_object[object_id] = []
                data_by_object[object_id].append(item)
        #Create sequences of features for each object
        sequences = []
        for object_id, data in data_by_object.items():
            if len(data) < seq_length:
                continue
            for i in range(len(data) - seq_length):
                seq = data[i:i + seq_length]
                sequences.append(seq)
        #Split data into train and test sets
        train_data, test_data = train_test_split(sequences, test_size=test_size, random_state=random_state)

        #Prepare data for model training
        train_X = np.array([seq[:-1] for seq in train_data])
        train_Y = np.array([seq[-1][3:6] for seq in train_data])
        test_X = np.array([seq[:-1] for seq in test_data])
        test_Y = np.array([seq[-1][3:6] for seq in test_data])

        #Normalize features
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
        test_X_Normalize = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

        #Reshape input for LSTM model
        train_X = np.reshape(train_X, (train_X.shape[0], seq_length-1, train_X.shape[2]))
        test_X_Normalize = np.reshape(test_X_Normalize, (test_X_Normalize.shape[0], seq_length-1, test_X_Normalize.shape[2]))

        return train_X, train_Y, test_X_Normalize, test_Y,test_X
