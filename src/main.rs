mod ann;
use std::{env, f32::INFINITY};

use ann::{Layer, ANN};
use ndarray::Array2;
fn main() {
    env::set_var("RUST_LOG", "debug");
    // env_logger::init();

    let mut layers: Vec<Layer> = Vec::new();
    layers.push(Layer {
        size: 64,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    layers.push(Layer {
        size: 100,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    layers.push(Layer {
        size: 100,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    layers.push(Layer {
        size: 100,
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

    ann.forward_propagation();
    let mut cost: f32 = INFINITY;

    let mut i = 0;
    while i < 1000000000 && cost > 0.0001  {
        i += 1;
        cost = ann.train(&expectation).unwrap();
        println!("{}",cost);
        ann.forward_propagation();
    }

    println!("Finished after {} tries!", i);
}
