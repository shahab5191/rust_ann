mod ann;
use std::{env, path::PathBuf, str::FromStr};

use ann::{Layer, ANN};
use ndarray::Array2;
use rust_ann::{
    array_from_dataframe, convert_data_to_image, convert_image_to_array, dataframe_from_csv, grayscale_image_data
};

fn main() {
    env::set_var("RUST_LOG", "debug");
    //env_logger::init();

    /*
    let mut layers = Vec::<Layer>::new();
    layers.push(Layer{size:4096, activation_function:ann::ActivationFunction::Sigmoid});
    layers.push(Layer{size:100, activation_function:ann::ActivationFunction::Sigmoid});
    layers.push(Layer{size:100, activation_function:ann::ActivationFunction::Sigmoid});
    layers.push(Layer{size:100, activation_function:ann::ActivationFunction::Sigmoid});
    layers.push(Layer{size:1, activation_function:ann::ActivationFunction::Sigmoid});
    let mut ann = ANN::new(layers, 0.005, 209, ann::CostFunction::BinaryCrossEntropy);
    let (data_set, labels) = dataframe_from_csv(
        PathBuf::from_str("./training_data/training_set.csv"
    ).unwrap()).unwrap();
    let raw_inputs: Array2<f32> = array_from_dataframe(&data_set);
    let inputs: Array2<f32> = grayscale_image_data(raw_inputs);
    let expected: Array2<f32> = array_from_dataframe(&labels);
    println!("Input size: {:?}", inputs.shape());
    println!("Expected Size: {:?}", expected.shape());
    let result = ann.train(inputs, &expected, 0.0005);
    match result {
        Ok(_) => println!("Model ran successfully!"),
        Err(e) => println!("Model returned error: {:?}", e)
    }
    match ann.save_model(PathBuf::from_str("./model.bin").unwrap()) {
        Ok(_) => println!("Model saved to file successfully!"),
        Err(e) => println!("{}", e)
    };
    */

    let mut ann = ANN::from_file(PathBuf::from_str("./model.bin").unwrap()).unwrap();
    let test_image = convert_image_to_array(PathBuf::from_str("./test_data/image_7.ppm").unwrap()).unwrap();
    let result = ann.predict(test_image);
    match result {
        Ok(r) => println!("image is {}% cat!", r * 100.0),
        Err(e) => println!("{e}"),
    }
}
