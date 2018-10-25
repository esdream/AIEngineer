import numpy as np
import sys


if sys.version_info.major == 2:
    input = raw_input


def read_matrix(N, M):
    matrix = np.empty((N, M), dtype=np.float32)
    for i in range(N):
        matrix[i, :] = list(map(float, input().rstrip().split(" ")))
    return matrix


def read_data():
    cur_row_idx = 0

    n_samples, input_size, hidden_size = map(int, input().rstrip().split(" "))
    cur_row_idx += 1

    X = read_matrix(
        n_samples,
        input_size)
    cur_row_idx += n_samples

    y = read_matrix(
        n_samples,
        1)
    cur_row_idx += n_samples

    input2hidden = read_matrix(
        input_size,
        hidden_size)
    cur_row_idx += input_size

    hidden2output = read_matrix(
        hidden_size,
        1)
    cur_row_idx += hidden_size
    return n_samples, input_size, hidden_size, X, y, input2hidden, hidden2output


def save_matrix(matrix):
    N, M = matrix.shape
    for i in range(N):
        print(" ".join(map("{:.2f}".format, matrix[i])))


def print_result(d_input2hidden, d_hidden2output):
    save_matrix(d_input2hidden)
    save_matrix(d_hidden2output)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, input2hidden, hidden2output):
        """
        Arguments
        ---------
        input_size (int):
            size of each input sample
        hidden_size (int):
            output size of the hidden layer
        input2hidden (ndarray): shape = [input_size, hidden_size]
            The initial weight for the hidden layer
        hidden2output (ndarray): shape = [hidden_size, 1]
            The initial weight for the output layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2hidden = input2hidden
        self.hidden2output = hidden2output

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def forward(self, X):
        """
        Arguments
        ---------
        X (ndarray): shape = [n_samples, input_size]
            the input

        Returns
        -------
        hidden (ndarray): shape = [n_samples, hidden_size]
            the output of the hidden layer
        output (ndarray): shape = [n_samples, 1]
            the output of the output layer
        """
        hidden = self.sigmoid(np.dot(X, self.input2hidden))
        output = self.sigmoid(np.dot(hidden, self.hidden2output))
        return hidden, output

    def backward(self, X, y, hidden, output):
        """
        Arguments
        ---------
        X (ndarray): shape = [n_samples, input_size]
            the input
        y (ndarray): shape = [n_samples, 1]
            the target
        hidden (ndarray): shape = [n_samples, hidden_size]
            the output of the hidden layer caculated by the forward pass
        output (ndarray): shape = [n_samples, 1]
            the output of the output layer caculated by the forward pass

        Returns
        -------
        d_input2hidden (ndarray): shape = [input_size, hidden_size]
            the gradient of the hidden layer's weights
        d_hidden2output (ndarray): shape = [hidden_size, 1]
            the gradient of the output layer's weights
        """
        ################################################
        # !!! Put your codes here and Fill this function
        
        # 这一行去掉：output_error_term = delta_output, hidden_error_term = delta_hidden, hidden_output = hidden, output = output

        delta_output = (y - output) * (output * (1.0 - output))
        hidden_error = np.dot(self.hidden2output, delta_output)
        delta_hidden = hidden_error * (hidden * (1.0 - hidden))
        ################################################
        d_input2hidden += np.dot(delta_hidden, X.T)  # modify these
        d_hidden2output += np.dot(delta_output, hidden.T)  # modify these
        return d_input2hidden, d_hidden2output


if __name__ == "__main__":
    n_samples, input_size, hidden_size, X, y, input2hidden, hidden2output = read_data()
    nn = NeuralNetwork(input_size, hidden_size, input2hidden, hidden2output)

    hidden, output = nn.forward(X)
    d_input2hidden, d_hidden2output = nn.backward(X, y, hidden, output)

    print_result(d_input2hidden, d_hidden2output)
