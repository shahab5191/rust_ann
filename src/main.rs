mod ann;
use std::{env, path::PathBuf, str::FromStr};

use ann::{Layer, ANN};
use ndarray::Array2;
use rust_ann::{array_from_dataframe, convert_data_to_image, dataframe_from_csv};

fn main() {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let res = dataframe_from_csv(PathBuf::from_str("./training_data/training_set.csv").unwrap()).unwrap();
    let arr = array_from_dataframe(&res.0);
    let res = convert_data_to_image(&arr);

    match res {
        Ok(_) => println!("Images saved!"),
        Err(e) => println!("Error: {:?}", e)
    };
}
