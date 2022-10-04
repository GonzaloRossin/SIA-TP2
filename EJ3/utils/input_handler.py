import numpy as np

class InputHandler:
    
    def random_layers_dim(self, input_dim, output_num, num_hidden_layers, max_dim):
        layers_dim = []
        layers_dim.append(input_dim)
        for i in range(num_hidden_layers):
            layers_dim.append(np.random.random_integers(2, max_dim))
        layers_dim.append(output_num)
        return layers_dim

    def __init__(self, input):

        self.apply_bias = (input['apply_bias']==1)
        self.num_layers = input['hidden_layers']['num_layers']
        self.learning_rate = input['learning_rate']

        self.model_type = input['model_type']
        self.hidden_activation = input['hidden_activation']
        self.output_activation = input['output_activation']

        if (input['from_file']==1):
            self.input_examples = np.genfromtxt(input['input_file'])
            training_idx = len(self.input_examples) * input['training_set_ratio'] // 100
            self.training_set = self.input_examples[:training_idx]
            self.test_set = self.input_examples[training_idx:]
        
        x_i_dim = input['single_input_dim']
        num_outputs = input['num_outputs']
        if (input['hidden_layers']['use_num']):
            self.layers_dim = self.random_layers_dim(x_i_dim, num_outputs, input['hidden_layers']['num_layers'], input['hidden_layers']['max_dim'])
        else:
            self.layers_dim = [x_i_dim] + input['hidden_layers']['layer_dims'] + [num_outputs]
        
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