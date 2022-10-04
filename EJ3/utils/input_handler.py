import numpy as np

class InputHandler:
    def __init__(self, input):
        self.input_examples = np.genfromtxt(input['input_file'])
        training_idx = len(self.input_examples) * input['training_set_ratio'] // 100
        self.training_set = self.input_examples[:training_idx]
        self.test_set = self.input_examples[training_idx:]

        self.apply_bias = (input['apply_bias']==1)
        self.num_layers = input['num_layers']
        self.num_iterations = input['num_iterations']
        self.learning_rate = input['learning_rate']

        self.model_type = input['model_type']
        self.hidden_activation = input['hidden_activation']
        self.output_activation = input['output_activation']

        self.use_mini_batches = (input['use_mini_batches']==1)
        if (self.use_mini_batches):
            self.batch_size = min(input['mini_batch_size'], len(self.training_set))
        else:
            self.batch_size = len(self.training_set)