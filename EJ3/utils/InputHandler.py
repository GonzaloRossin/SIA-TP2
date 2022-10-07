import numpy as np
from utils.constants import LOGISTIC
from multilayer_utils.Normalization import normalize

class InputHandler:
    
    def random_layers_dim(self, feature_num, output_num, num_hidden_layers, max_dim):
        layers_dim = []
        layers_dim.append(feature_num)
        for i in range(num_hidden_layers):
            layers_dim.append(np.random.random_integers(2, max_dim))
        layers_dim.append(output_num)
        return layers_dim

    def read_input(self, input_filepath, num_features):
        X = []
        Y = []
        with open(input_filepath) as input_file:
            input_file.readline()   # skip header
            lines = input_file.readlines()
            for line in lines:
                XY = line.split(",")
                XY = list(map(lambda z: int(z), XY))
                X.append(XY[:num_features])
                Y.append(XY[num_features:])
        return X, Y

    def __init__(self, input):

        self.apply_bias = (input['apply_bias']==1)
        self.num_layers = input['hidden_layers']['num_layers']
        self.learning_rate = input['learning_rate']

        self.normalize = (input['normalize']==1)
        self.hidden_activation = input['hidden_activation']
        self.output_activation = input['output_activation']

        num_features = input['num_features']
        num_outputs = input['num_outputs']

        X, Y = self.read_input(input['input_file'], num_features)
        if (self.normalize):
            Y, self.min_y, self.max_y = normalize(Y, self.output_activation)

        self.ratio = input['training_set_ratio']
        training_idx = len(X) * self.ratio // 100
        train_X = X[:training_idx]
        train_Y = Y[:training_idx]
        test_X = X[training_idx:]
        test_Y = Y[training_idx:]

        self.training_set_X = np.array(train_X).T
        self.test_set_X = np.array(test_X).T
        self.training_set_Y = np.array(train_Y).T
        self.test_set_Y = np.array(test_Y).T
        
        if (input['hidden_layers']['use_num']):
            self.layers_dim = self.random_layers_dim(num_features, num_outputs, input['hidden_layers']['num_layers'], input['hidden_layers']['max_dim'])
        else:
            self.layers_dim = [num_features] + input['hidden_layers']['layer_dims'] + [num_outputs]
        print(f"\nLayers dim = {self.layers_dim}\n")

        self.num_epochs = input['num_epochs']
        # TODO: Implement mini-batches
        
        '''
        self.use_mini_batches = (input['use_mini_batches']==1)
        if (self.use_mini_batches):
            self.batch_size = min(input['mini_batch_size'], len(self.training_set))
        else:
            self.batch_size = len(self.training_set)
        '''