use std::env;
use std::error::Error;
use crate::nn::model::Model;

mod nn;
mod mnist_dataset;


fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv()?;

    let train = mnist_dataset::load_mnist_dataset(env::var("MNIST_TRAIN_DATA")?.as_str())?;
    let test = mnist_dataset::load_mnist_dataset(env::var("MNIST_TEST_DATA")?.as_str())?;

    let train_dataset = nn::utils::normalize(&train);
    let test_dataset = nn::utils::normalize(&test);

    let model_config = nn::model::ModelConfig {
        optimizer: Box::new(nn::optimizer::AdamOptimizer),
        loss: Box::new(nn::loss::SparseCategoricalCrossentropyLoss),
    };
    let mut model = nn::model::SequentialModel::new(model_config);
    model.add_layer(Box::new(nn::layer::DenseLayer::new(784, Box::new(nn::activation::ReLU)))); // 28 * 28 is already flattened
    model.add_layer(Box::new(nn::layer::DenseLayer::new(128, Box::new(nn::activation::ReLU))));
    model.add_layer(Box::new(nn::layer::DenseLayer::new(128, Box::new(nn::activation::ReLU))));
    model.add_layer(Box::new(nn::layer::DenseLayer::new(10, Box::new(nn::activation::Softmax))));

    model.fit(&train_dataset, 10, 0.001);

    let prediction = model.predict(test_dataset.inputs.get(0).unwrap());
    println!("Prediction: {:?}", prediction);

    println!("Hello, world!");

    Ok(())
}
