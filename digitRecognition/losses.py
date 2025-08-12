import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentrophy(Loss): 
    CategoricalCrossentrophyInstances = []
    def __init__(self):
        Loss_CategoricalCrossentrophy.CategoricalCrossentrophyInstances.append(self)

    def forward(self, predictions, targets):
        self.samples = len(predictions)
        self.predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        self.targets = targets

        if len(targets.shape) == 1:
            correct_confidences = self.predictions_clipped[range(self.samples), targets]
        elif len(targets.shape) == 2:
            correct_confidences = np.sum(self.predictions_clipped * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self):
        # Use stored clipped predictions and targets
        if len(self.targets.shape) == 1:
            one_hot_targets = np.zeros_like(self.predictions_clipped)
            one_hot_targets[range(self.samples), self.targets] = 1
            delta = -one_hot_targets / self.predictions_clipped
        elif len(self.targets.shape) == 2:
            delta = -self.targets / self.predictions_clipped

        delta = delta / self.samples
        return delta.T

class MSELoss(Loss): 
    diff = None
    def forward(self, predictions, targets):
        predictions = np.array(predictions)
        targets = np.array(targets)

        MSELoss.diff = predictions - targets
        mse = np.mean(self.diff ** 2)
        return mse
    @staticmethod
    def backward():
        return (2 * MSELoss.diff / len(MSELoss.diff)).T
