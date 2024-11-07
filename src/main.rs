mod ann;
use std::{env, f32::INFINITY};

use ann::{Layer, ANN};
use log::info;
use ndarray::Array2;
fn main() {
    env::set_var("RUST_LOG", "debug");
    //env_logger::init();

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
    let mut ann = ANN::new(layers, 100.0, 1, ann::CostFunction::MeanSquaredError);

    ann.initialize_parameters();
    let expectation_vec = vec![1.0, 0.5];
    let expectation: Array2<f32> = Array2::from_shape_vec((2, 1), expectation_vec).unwrap();

    ann.forward_propagation();
    let mut cost: f32 = INFINITY;

    let mut i = 0;
    while cost > 0.01 && i < 1000 {
        i += 1;
        cost = ann.train(&expectation).unwrap();
        println!("{}",cost);
        ann.forward_propagation();
    }
}
