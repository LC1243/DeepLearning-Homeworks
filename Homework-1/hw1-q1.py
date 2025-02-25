#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        
        # calculate predicted y
        y_hat = self.predict(x_i)
        eta = kwargs.get("learning_rate", 1)
        
        # update weights if prediction is wrong
        if (y_hat != y_i):
            self.W[y_i,:] += eta * x_i
            self.W[y_hat,:] -= eta * x_i
        #raise NotImplementedError

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.2a

        # Get probability scores according to the model (num_labels x 1).
        label_scores = np.expand_dims(self.W.dot(x_i), axis=1)

        # One-hot encode the true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        # Compute softmax probabilities (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))

        # Compute the gradient from the loss (without reg)
        gradient = (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis=1).T)
       
        # Add L2 regularization to the gradient if l2_penalty > 0
        if l2_penalty > 0.0:
            regularization_gradient = l2_penalty * self.W
            gradient -= regularization_gradient
        
        # Update the weights using gradient descent
        self.W += learning_rate * gradient


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.W = [np.random.normal(0.1, 0.1, (hidden_size, n_features)),
                  np.random.normal(0.1, 0.1, (n_classes, hidden_size))]
        
        self.b = [np.zeros(hidden_size), np.zeros(n_classes)]
        # raise NotImplementedError # Q1.3 (a)

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        # Numerically stable softmax (subtraction of max(x) to avoid overflows)
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    
    #Assuming y is one-hot encoded and y_hat is the output of the last layer
    def compute_loss(self, y_hat, y):
        probs = self.softmax(y_hat)
        epsilon = 1e-12
        loss = -y.dot(np.log(probs + epsilon))
        return loss 
    
    def forward_pass(self, x):
        num_layers = len(self.W)
        hidden_layers = []

        for i in range(num_layers):
            h = x if i == 0 else hidden_layers[i - 1]
            z = self.W[i].dot(h) + self.b[i]
            
            #Apply relu activation to hidden layers
            if i < num_layers - 1:
                hidden_layers.append(self.relu(z))
        
        #z is the output of the last layer before activation function
        return z, hidden_layers
    

    def backward_propagation(self, x, y, hidden_layers, z):
        num_layers = len(self.W)

        # Compute the gradient of the loss with respect to the output of the last layer
        probs = self.softmax(z)
        gradient_z = probs - y

        gradient_weights, gradient_biases = [], []

        for i in range(num_layers - 1, -1, -1):

            # Gradient of hidden layer weights and biases
            h = x if i == 0 else hidden_layers[i - 1]
            gradient_weights.append(gradient_z[:, None].dot(h[:, None].T))
            gradient_biases.append(gradient_z)

            # Gradient of hidden layer below.
            gradient_h = self.W[i].T.dot(gradient_z)
            
            # Gradient of hidden layer below before activation.
            gradient_z = gradient_h * self.relu_derivative(h)

        # Reverse the lists before returning
        gradient_weights.reverse()
        gradient_biases.reverse()    

        return gradient_weights, gradient_biases     

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        # Q1.3 (a)
        
        predictions = []
        for x in X:
            #Forward pass and get the output (class with highest probability)
            z, _ = self.forward_pass(x)
            y_hat = np.argmax(z)
            predictions.append(y_hat)

        predictions = np.array(predictions)
        return predictions

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        # Q1.3 (a)
        num_layers = len(self.W)
        total_loss = 0
        
        # For each observation and target
        for x_i, y_i in zip(X, y):

            output, hidden_layers = self.forward_pass(x_i)
            
            # Compute Loss and Update total loss
            y_one_hot = np.zeros(output.shape)
            y_one_hot[y_i] = 1

            loss = self.compute_loss(output, y_one_hot)
            total_loss += loss
            
            # Compute back propagation
            grad_weights, grad_biases = self.backward_propagation(x_i, y_one_hot, hidden_layers, output)
            
            # Update weights
            for i in range(num_layers):
                self.W[i] -= learning_rate * grad_weights[i]
                self.b[i] -= learning_rate * grad_biases[i]
                
        return total_loss
        # raise NotImplementedError 

def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
