mod ann;
use std::env;

use ann::{Layer, ANN};
use ndarray::Array2;
fn main() {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let mut layers: Vec<Layer> = Vec::new();
    layers.push(Layer {
        size: 2,
        activation_function: ann::ActivationFunction::Relu,
    });
    layers.push(Layer {
        size: 2,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    layers.push(Layer {
        size: 2,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    let mut ann = ANN::new(layers, 0.01, 1);

    ann.initialize_parameters();
    let expectation_vec = vec![1.0, 0.5];
    let expectation: Array2<f32> = Array2::from_shape_vec((2, 1), expectation_vec).unwrap();
    println!("Calling forward propagation");
    ann.forward_propagation();
    ann.print_layers();
}
