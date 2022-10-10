class InputHandler:
    def __init__(self, input):
        self.operation = input['operation']
        self.learning_rate = input['learning_rate']
        self.num_epochs = input['num_epochs']