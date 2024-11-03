mod ann;
use std::{path::PathBuf, str::FromStr, time::Instant};

use ann::{Layer, ANN};
fn main() {
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
    ann.save_model(PathBuf::from_str("./test.bin").unwrap())
        .unwrap_or_else(|err| println!("{}", err));
    ann.from_file(PathBuf::from_str("./test.bin").unwrap())
        .unwrap_or_else(|err| println!("{}", err));
    ann.initialize_parameters();
    let start = Instant::now();
    ann.forward_propagation();
    let duration = start.elapsed();
    println!("Elapsed time: {:?}", duration);
}
