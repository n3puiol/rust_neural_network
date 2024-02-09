use std::env;
use std::error::Error;
use crate::nn::model::Model;

mod nn;
mod mnist_dataset;


fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv()?;

    let train = mnist_dataset::load_mnist_dataset(env::var("MNIST_TRAIN_DATA")?.as_str(), 5)?;
    let test = mnist_dataset::load_mnist_dataset(env::var("MNIST_TEST_DATA")?.as_str(), 5)?;

    let train_dataset = nn::utils::normalize(&train);
    let test_dataset = nn::utils::normalize(&test);

    let model_config = nn::model::ModelConfig {
        optimizer: Box::new(nn::optimizer::GradientDescentOptimizer),
        loss: Box::new(nn::loss::MeanSquaredErrorLoss),
    };
    let mut model = nn::model::SequentialModel::new(model_config);
    model.add_layer(Box::new(nn::layer::InputLayer::new())); // 28 * 28 is already flattened
    model.add_layer(Box::new(nn::layer::FullyConnectedLayer::new(784, 128, Box::new(nn::activation::ReLU))));
    model.add_layer(Box::new(nn::layer::FullyConnectedLayer::new(128, 128, Box::new(nn::activation::ReLU))));
    model.add_layer(Box::new(nn::layer::FullyConnectedLayer::new(128, 10, Box::new(nn::activation::Sigmoid))));

    model.fit(&train_dataset, 20, 0.001);

    println!("{:?}", test_dataset.targets.get(1));
    let prediction = model.predict(test_dataset.inputs.get(1).unwrap());
    println!("Prediction: {:?}", prediction);

    println!("Hello, world!");

    Ok(())
}
