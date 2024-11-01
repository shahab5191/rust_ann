use std::path::PathBuf;
use ndarray::Array2;
use polars::prelude::*;

mod ann;

pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {
    let data = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;
    Ok((training_dataset, training_labels))
}

pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::C).unwrap().reversed_axes()
}
