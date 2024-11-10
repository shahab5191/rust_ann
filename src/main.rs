mod ann;
use std::{env, f32::INFINITY, path::PathBuf, str::FromStr};

use ann::{Layer, ANN};
use ndarray::Array2;
fn main() {
    env::set_var("RUST_LOG", "debug");
    // env_logger::init();

    let mut layers: Vec<Layer> = Vec::new();
    layers.push(Layer {
        size: 3,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    layers.push(Layer {
        size: 4,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    layers.push(Layer {
        size: 3,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    let mut ann = ANN::new(layers, 0.01, 1, ann::CostFunction::MeanSquaredError);

    ann.initialize_parameters();
    let expectation_vec = vec![1.0, 0.5, 0.0];
    let expectation: Array2<f32> = Array2::from_shape_vec((3, 1), expectation_vec).unwrap();

    ann.print_layers();
    let _ = ann.save_model(PathBuf::from_str("./test.data").unwrap());

    let ann2 = ANN::from_file(PathBuf::from_str("./test.data").unwrap()).unwrap();
    ann2.print_layers();
    
}
