from dataclasses import dataclass
from typing import Tuple

from util import *
import numpy as np


@dataclass
class BatchResult:
    loss: np.double
    correct: any


class MultiLayerNetWork:
    """
    a three layer network
    input layer
    hidden layer: activation function is sigmoid
    output layer: activation function is softmax
    loss function: cross_entropy

    use the backward propagation with cross along with minimatch train method
    adjust weight in the network
    """

    def __init__(self, n_input, n_hidden, n_output, learning_rate, n_epoch,
                 batch_size, bias=0, init=True):
        if init:
            self.n_epoch = n_epoch
            self.batch_size = batch_size
            self.n_hidden = n_hidden
            self.epoch = 0
            self.bias = bias
            self.learning_rate = learning_rate
            self.hidden_weight = np.zeros((n_input, n_hidden))
            self.output_weight = np.zeros((n_hidden, n_output))
            self.hidden_bias = np.zeros(n_hidden)
            self.output_bias = np.zeros(n_output)

    @classmethod
    def load_net_work(cls, serial_number):
        """
        load the network from file
        :param serial_number:
        :return:
        """
        with open(f"savings/train_saving{serial_number}.json", "r") as fp:
            j = json.load(fp)
            ins = cls(None, None, None, None, None, None, init=False)
            ins.n_epoch = j["n_epoch"]
            ins.batch_size = j["batch_size"]
            ins.epoch = j["epoch"]
            ins.bias = j["bias"]
            ins.learning_rate = j["learning_rate"]
            ins.hidden_bias = np.asarray(j["hidden_bias"], dtype=np.double)
            ins.output_bias = np.asarray(j["output_bias"], dtype=np.double)
            ins.output_weight = np.asarray(j["output_weight"], dtype=np.double)
            ins.hidden_weight = np.asarray(j["hidden_weight"], dtype=np.double)
            return ins

    def save_net_work(self, serial_number):
        """
        saving the network
        :param serial_number:
        """
        with open(f"savings/train_saving{serial_number}.json", "w") as fp:
            json.dump({
                "n_hidden": self.n_hidden,
                "batch_size": self.batch_size,
                "n_epoch": self.n_epoch,
                "epoch": self.epoch,
                "bias": self.bias,
                "learning_rate": self.learning_rate,
                "hidden_weight": self.hidden_weight,
                "output_weight": self.output_weight,
                "hidden_bias": self.hidden_bias,
                "output_bias": self.output_bias,
            }, fp, cls=NpEncoder)

    def forward(self, inputs) -> Tuple[np.ndarray, np.ndarray]:
        hidden_output = sigmoid(inputs.dot(self.hidden_weight) + self.hidden_bias * self.bias)
        output = softmax(hidden_output.dot(self.output_weight) + self.output_bias * self.bias)
        return hidden_output, output

    def forward_backward_batch(self, inputs, targets, debug=False) -> BatchResult:
        """
        forward propagate a batch inputs and get output with softmax
        :param debug: if debug printout process
        :param targets: expected
        :param inputs: initial input to the net work
        :return: BatchResult containing train information
        """
        # forward
        hidden_output, output = self.forward(inputs)
        correct_count = np.sum(targets.argmax(axis=1) == output.argmax(axis=1))
        loss = cross_entropy(output, targets)

        # backward
        delta_output_tot = output - targets
        delta_output = hidden_output.T.dot(delta_output_tot)
        delta_hidden_tot = delta_output_tot.dot(self.output_weight.T)
        sig_der = sigmoid_derivative(hidden_output)
        delta_hidden = inputs.T.dot(sig_der * delta_hidden_tot)

        # update construction
        output_bias_update = self.learning_rate * delta_output_tot.sum(axis=0)  # sum over all samples
        hidden_bias_update = self.learning_rate * (delta_hidden_tot * sig_der).sum(axis=0)  # sum over all samples
        output_update = self.learning_rate * delta_output
        hidden_update = self.learning_rate * delta_hidden

        self.output_weight -= output_update
        self.hidden_weight -= hidden_update
        self.output_bias -= output_bias_update
        self.hidden_bias -= hidden_bias_update
        return BatchResult(loss, correct_count)

    def one_hot_predict(self, inputs) -> np.ndarray:
        """
        :param inputs: data
        :return one shot encoded in numpy array
        """
        output = self.predict(inputs)
        encoded = np.zeros(output.shape)
        indics = output.argmax(axis=1)
        encoded[np.arange(indics.size), indics] = 1
        return encoded

    def predict(self, inputs):
        """
        return a predict output
        :param inputs:
        :return:
        """
        _, output = self.forward(inputs)
        return output

    def __str__(self):
        s = ""
        s += "output_weight: " + str(list(list(a) for a in self.output_weight)) + str(self.output_bias) + "\n"
        s += "hidden_weight: " + str(list(list(a) for a in self.hidden_weight)) + str(self.hidden_bias) + "\n"
        return s

    def __repr__(self):
        return str(self)

    def train_from_gen(self, train_data, train_label, test_data, test_label):
        """
        train the network with mini patch method
        and saving result to file
        :param self: network
        :param train_data: train_data
        :param train_label: train_label
        :param test_label: test_label
        :param test_data: test_data
        """
        total = 0
        for epoch in range(self.n_epoch):
            batch_count = 0
            total_loss = 0
            for batch_data, batch_label in zip(group_n(train_data, self.batch_size),
                                               group_n(train_label, self.batch_size)):
                total += self.batch_size
                batch_count += 1
                result = self.forward_backward_batch(batch_data, batch_label)
                total_loss += result.loss
                if batch_count % 30 == 0:
                    print(
                        f"epoch={epoch}, batch={batch_count}, loss: {result.loss}, rate: {result.correct / len(batch_data)}")
            if epoch % 5 == 0:
                cross_correct_rate = self.cross_validate(test_data, test_label)
                self.save_net_work(int(cross_correct_rate * 100))
        print("train is done")

    def cross_validate(self, data, label):
        """
        validate the model with test data
        :param self: model
        :param data:
        :param label:
        :return: correct rate of data
        """
        count = len(label)
        output = self.predict(data)
        correct_count = np.sum(label.argmax(axis=1) == output.argmax(axis=1))
        loss = cross_entropy(output, label) / (count + 1)
        correct = correct_count
        print(f"correct: {correct} out of {count}, rate: {correct / (count + 1)}, loss: {loss}")
        if count > 0:
            return correct / count
        return 0


def group_n(gen, n):
    """
    chunks data into groups of size n
    :param gen: data source
    :param n: group size
    """
    data = []
    count = 0
    for row in gen:
        count += 1
        data.append(np.asarray(row, dtype=float))
        if count == n:
            yield np.asarray(data)
            data = []
            count = 0


def test():
    a = np.asarray([1, 2])
    b = np.asarray([[1, 2], [2, 3]])
    print(b @ a)
    print(b * a)


if __name__ == "__main__":
    net = MultiLayerNetWork(n_input=2, n_hidden=2, n_output=1, bias=-1, learning_rate=0.2, n_epoch=0, batch_size=1)
    xs = np.asarray([[.3, .4], [.1, .6], [.9, .4]])
    ts = np.asarray([[0.88], [0.82], [0.57]])
    net.save_net_work(11)
    print(net.forward(xs))
    print(net.forward_backward_batch(xs, ts))
