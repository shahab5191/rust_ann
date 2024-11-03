mod ann;
use std::{path::PathBuf, str::FromStr, time::Instant};

use ann::{Layer, ANN};
fn main() {
    /*
    let mut layers: Vec<Layer> = Vec::new();
    layers.push(Layer {
        size: 2,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    layers.push(Layer {
        size: 3,
        activation_function: ann::ActivationFunction::Sigmoid,
    });
    let mut ann = ANN::new(layers, 0.01, 1);
    
    ann.initialize_parameters();
    */
    let ann = ANN::from_file(PathBuf::from_str("./test.bin").unwrap()).unwrap();
    ann.print_layers();
}
