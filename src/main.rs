mod ann;
use std::{env, path::PathBuf, str::FromStr};

use ann::{Layer, ANN};
use ndarray::Array2;
use rust_ann::{array_from_dataframe, dataframe_from_csv, grayscale_image_data};

fn main() {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();
    let mut layers = Vec::<Layer>::new();
    layers.push(Layer{size:4096, activation_function:ann::ActivationFunction::Sigmoid});
    layers.push(Layer{size:100, activation_function:ann::ActivationFunction::Relu});
    layers.push(Layer{size:100, activation_function:ann::ActivationFunction::Sigmoid});
    layers.push(Layer{size:1, activation_function:ann::ActivationFunction::Sigmoid});
    let mut ann = ANN::new(layers, 1.0, 209, ann::CostFunction::BinaryCrossEntropy);
    let (data_set, labels) = dataframe_from_csv(
        PathBuf::from_str("./training_data/training_set.csv"
    ).unwrap()).unwrap();
    let raw_inputs: Array2<f32> = array_from_dataframe(&data_set);
    let inputs: Array2<f32> = grayscale_image_data(raw_inputs);
    let expected: Array2<f32> = array_from_dataframe(&labels);
    println!("Input size: {:?}", inputs.shape());
    println!("Expected Size: {:?}", expected.shape());
    let result = ann.train(inputs, &expected, 0.0001);
    match result {
        Ok(_) => println!("Model ran successfully!"),
        Err(e) => println!("Model returned error: {:?}", e)
    }
}
