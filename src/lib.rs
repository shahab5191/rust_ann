use log::info;
use ndarray::Array2;
use polars::prelude::*;
use core::f32;
use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf, usize,
};
mod ann;

#[derive(Debug, Clone, Copy)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
}

pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {
    let data: DataFrame = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;
    Ok((training_dataset, training_labels))
}

pub fn array_from_dataframe(df: &DataFrame) -> Array2<u8> {
    let arr = df
        .to_ndarray::<Int64Type>(IndexOrder::C)
        .unwrap()
        .reversed_axes();
    arr.mapv(|x| {
        if x < 0 {
            0
        } else if x > 255 {
            255
        } else {
            x as u8
        }
    })
}

pub fn convert_data_to_image(data_set: &Array2<u8>) -> Result<(), io::Error> {
    let image_count = data_set.shape()[1];
    let image_binary_len = data_set.shape()[0];

    info!("DataSet rows count: {}", image_count);
    info!("Image binary length {:?}", image_binary_len);

    if image_binary_len % 3 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Data set shape is not 3 channel color!\nBinary length: {}", image_binary_len),
        ));
    }

    let image_dim_sqrt = f32::sqrt(image_binary_len as f32 / 3.0);
    if image_dim_sqrt != f32::floor(image_dim_sqrt) {
        println!("image dim sqrt: {}, floor: {}", image_dim_sqrt, f32::floor(image_dim_sqrt));
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Images must be square!",
        ));
    }

    let image_dim = f32::sqrt(image_binary_len as f32/ 3.0) as usize;

    let mut images: Vec<Array2<Color>> = Vec::new();

    info!("Converting raw data into Array2 of color");

    for col in data_set.columns() {
        let image_binary_vector: Vec<u8> = col.to_vec();
        let mut image_vec: Vec<Color> = Vec::new();
        for i in 0..(image_binary_vector.len() / 3) {
            let color: Color = Color {
                r: image_binary_vector[i * 3],
                g: image_binary_vector[i * 3 + 1],
                b: image_binary_vector[i * 3 + 2],
            };

            image_vec.push(color);
        }
        let image_array: Array2<Color> =
            Array2::from_shape_vec((image_dim, image_dim), image_vec).unwrap();
        images.push(image_array);
    }

    save_image_to_ppm(images, PathBuf::from("./"))?;

    Ok(())
}

fn save_image_to_ppm(data: Vec<Array2<Color>>, path: PathBuf) -> Result<(), io::Error> {
    info!("Saving images!");
    let height = data[0].shape()[0];

    let header = format!("P6\n{} {}\n255\n", height, height);

    for i in 0..data.len() {
        let file_name = path.join(format!("{i}.ppm"));
        info!("Image number {i} filename: {:?}", file_name);
        let mut file = File::create(file_name)?;
        file.write_all(header.as_bytes())?;

        for row in data[i].rows() {
            for &Color { r, g, b } in row.iter() {
                file.write_all(&[r, g, b])?;
            }
        }
    }
    Ok(())
}
