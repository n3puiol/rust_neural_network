# [WIP] Simple Artificial Neural Network written in Rust

This is a simple implementation of an artificial neural network written in Rust. The goal of this project is to learn
about the inner workings of neural networks and to get a better understanding of the Rust programming language.

No libraries are used for the neural network implementation, only the standard library of Rust.

An example using the MNIST dataset is provided to demonstrate the usage of the neural network.

## Usage

To run the example, you need to download the MNIST dataset and extract it into the `data` directory. The CSV dataset can
be
downloaded from the following link: [MNIST dataset](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/).

### .env file

Create a `.env` file in the root directory of the project and add the following content:

```env
MNIST_TRAIN_DATA="<path to the training data>"
MNIST_TEST_DATA="<path to the test data>"
```

### Run the example

```bash
cargo run --color=always --package rust-neural-network --bin rust-neural-network
```