mod ann;
use std::env;

use ann::{Layer, ANN};
use ndarray::Array2;

fn main() {
    env::set_var("RUST_LOG", "debug");
    //env_logger::init();
    let mut layers = Vec::<Layer>::new();
    layers.push(Layer{size:2, activation_function:ann::ActivationFunction::Sigmoid});
    layers.push(Layer{size:3, activation_function:ann::ActivationFunction::Sigmoid});
    layers.push(Layer{size:2, activation_function:ann::ActivationFunction::Sigmoid});
    let mut ann = ANN::new(layers, 0.01, 2, ann::CostFunction::MeanSquaredError);
    let inputs_vec: Vec<f32> = vec![0.2, 0.1, 0.3, 0.4];
    //let inputs_vec: Vec<f32> = vec![0.2, 0.1];
    let inputs: Array2<f32> = Array2::from_shape_vec((2,2), inputs_vec).unwrap();
    let expected_vec: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    //let expected_vec: Vec<f32> = vec![1.0, 0.0];
    let expected: Array2<f32> = Array2::from_shape_vec((2,2), expected_vec).unwrap();
    let result = ann.train(inputs, &expected, 0.0001);
    match result {
        Ok(_) => println!("Model ran successfully!"),
        Err(e) => println!("Model returned error: {:?}", e)
    }
}
