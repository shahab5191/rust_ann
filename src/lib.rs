use core::f32;
use log::info;
use ndarray::{s, Array1, Array2};
use polars::prelude::*;
use std::{
    fs::File,
    io::{self, BufRead, BufReader, Read, Write},
    path::PathBuf,
    usize,
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

    let training_labels = data.select_at_idx(0).unwrap().clone().into_frame();
    let col_names = data.get_column_names();
    let remaining_columns: Vec<&str> = col_names[1..].iter().map(|x| x.as_ref()).collect();
    let training_dataset = data.select(remaining_columns).unwrap();
    Ok((training_dataset, training_labels))
}

pub fn expand_labels(col: DataFrame) -> Result<Array2<f32>, io::Error> {
    let col_size = col.shape().0;
    let label_array = array_from_dataframe(col);
    let mut max: f32 = 0.0;
    for i in &label_array {
        if *i > max {
            max = *i;
        }
    }
    max += 1.0;
    let mut expanded_vec: Vec<f32> = vec![0.0; col_size * max as usize];
    for (i, label) in label_array.iter().enumerate() {
        expanded_vec[i * max as usize + *label as usize] = 1.0;
    }
    let expanded_array: Array2<f32> =
        Array2::<f32>::from_shape_vec((max as usize, col_size), expanded_vec).map_err(
            |_| {
                io::Error::new(
                    io::ErrorKind::Other,
                    "Error converting expanded vec to array2",
                )
            },
        )?;
    Ok(expanded_array)
}

pub fn array_from_dataframe(df: DataFrame) -> Array2<f32> {
    let arr = df
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap()
        .reversed_axes();
    arr
}

pub fn grayscale_image_data(images: Array2<f32>) -> Array2<f32> {
    let expected_row_size = images.dim().0 / 3;
    let col_size = images.dim().1;
    println!("Expected row size: {expected_row_size}, col size: {col_size}");
    let mut new_images: Array2<f32> = Array2::<f32>::zeros((expected_row_size, col_size));
    println!("new images size: {:?}", new_images.shape());

    for (i, col) in images.columns().into_iter().enumerate() {
        let image_vec = col.to_vec();
        let mut grayscale_vec: Vec<f32> = Vec::new();
        for i in 0..image_vec.len() / 3 {
            let pixel = (image_vec[i * 3] + image_vec[i * 3 + 1] + image_vec[i * 3 + 2]) / 3.0;
            grayscale_vec.push(pixel);
        }
        new_images
            .slice_mut(s![.., i])
            .assign(&Array1::from(grayscale_vec));
    }

    new_images / 255.0
}

pub fn convert_data_to_image(data_set: &Array2<f32>) -> Result<(), io::Error> {
    let image_count = data_set.shape()[1];
    let image_binary_len = data_set.shape()[0];

    info!("DataSet rows count: {}", image_count);
    info!("Image binary length {:?}", image_binary_len);

    if image_binary_len % 3 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Data set shape is not 3 channel color!\nBinary length: {}",
                image_binary_len
            ),
        ));
    }

    let image_dim_sqrt = f32::sqrt(image_binary_len as f32 / 3.0);
    if image_dim_sqrt != f32::floor(image_dim_sqrt) {
        println!(
            "image dim sqrt: {}, floor: {}",
            image_dim_sqrt,
            f32::floor(image_dim_sqrt)
        );
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Images must be square!",
        ));
    }

    let image_dim = f32::sqrt(image_binary_len as f32 / 3.0) as usize;

    let mut images: Vec<Array2<Color>> = Vec::new();

    info!("Converting raw data into Array2 of color");

    for col in data_set.columns() {
        let image_binary_vector: Vec<f32> = col.to_vec();
        let mut image_vec: Vec<Color> = Vec::new();
        for i in 0..(image_binary_vector.len() / 3) {
            let color: Color = Color {
                r: image_binary_vector[i * 3] as u8,
                g: image_binary_vector[i * 3 + 1] as u8,
                b: image_binary_vector[i * 3 + 2] as u8,
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

pub struct PpmImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub fn convert_image_to_array(file_path: PathBuf) -> Result<Array2<f32>, io::Error> {
    let image: PpmImage = PpmImage::from_file(file_path)?;
    if image.height != image.width && image.height != 64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Image size is not correct",
        ));
    }
    let image_binary_length: usize = (image.height * image.width * 3) as usize;
    let image_data_float = image.data.into_iter().map(|x| x as f32).collect();
    let image_data =
        Array2::<f32>::from_shape_vec((image_binary_length, 1), image_data_float).unwrap();
    let grayscale_image = grayscale_image_data(image_data);
    Ok(grayscale_image)
}

impl PpmImage {
    pub fn from_file(file_path: PathBuf) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        // Read and validate the PPM header
        let mut header = String::new();
        reader.read_line(&mut header)?; // Read the magic number
        if header.trim() != "P6" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid PPM magic number",
            ));
        }

        // Skip comments if present
        let mut width = String::new();
        loop {
            width.clear();
            reader.read_line(&mut width)?;
            if !width.starts_with('#') {
                break;
            }
        }

        // Parse width and height
        let dims: Vec<&str> = width.trim().split_whitespace().collect();
        let width = dims[0]
            .parse::<u32>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid width"))?;
        let height = dims[1]
            .parse::<u32>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid height"))?;

        // Read max color value (typically 255)
        let mut max_val = String::new();
        reader.read_line(&mut max_val)?;
        if max_val.trim() != "255" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported max color value",
            ));
        }

        // Read pixel data
        let mut data = vec![0; (width * height * 3) as usize];
        reader.read_exact(&mut data)?;

        Ok(PpmImage {
            width,
            height,
            data,
        })
    }
}
