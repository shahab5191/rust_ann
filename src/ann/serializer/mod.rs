use std::{
    io::{self, Cursor, ErrorKind, Read},
    usize,
};

use ndarray::{Array2, Axis};

use crate::ann::{ActivationFunction, ANN};

use super::{CostFunction, Layer};
pub struct Serializer {}

impl Serializer {
    fn serialize_layers(&self, vec: &Vec<Layer>) -> Vec<u8> {
        let mut buffer: Vec<u8> = Vec::new();
        buffer.extend((vec.len() as u32).to_le_bytes());
        let mut layers_buffer: Vec<u8> = Vec::new();
        for l in vec {
            layers_buffer.extend((l.size as u32).to_le_bytes());
            let act_func: u8 = l.activation_function.clone() as u8;
            layers_buffer.extend(act_func.to_le_bytes());
        }
        buffer.extend((layers_buffer.len() as u32).to_le_bytes());
        buffer.extend(layers_buffer);
        buffer
    }

    fn serialize_vec(&self, vec: &Vec<Array2<f32>>) -> Vec<u8> {
        let mut buffer: Vec<u8> = Vec::new();
        buffer.extend((vec.len() as u32).to_le_bytes());
        let mut act_vec: Vec<u8> = Vec::new();
        for act_mat in vec {
            let mut array_bytes: Vec<u8> = Vec::new();
            let cols: usize = act_mat.len_of(Axis(0));
            let rows: usize = act_mat.len_of(Axis(1));
            array_bytes.extend((cols as u32).to_le_bytes());
            array_bytes.extend((rows as u32).to_le_bytes());
            for a in act_mat {
                array_bytes.extend(a.to_le_bytes());
            }
            act_vec.extend((array_bytes.len() as u32).to_le_bytes());
            act_vec.extend(array_bytes);
        }
        buffer.extend((act_vec.len() as u32).to_le_bytes());
        buffer.extend(act_vec);

        println!("buffer: {:?}", buffer);

        buffer
    }

    pub fn serialize(&self, ann: &ANN) -> Result<Vec<u8>, std::io::Error> {
        let mut buffer: Vec<u8> = Vec::new();

        // Adding Header
        let header = "SHIRIN_ANN".as_bytes();
        buffer.extend(header);

        // Serializing layers
        buffer.extend(self.serialize_layers(&ann.layers));

        // Serializing learning_rate
        buffer.extend(ann.learning_rate.to_le_bytes());

        // Serializing example_number
        buffer.extend((ann.example_number as u32).to_le_bytes());

        // Serializing cost_function
        buffer.extend((ann.cost_function.clone() as u8).to_le_bytes());

        // Serializig logit_matrices
        buffer.extend(self.serialize_vec(&ann.logit_matrices));

        // Serializing activation_matrices
        buffer.extend(self.serialize_vec(&ann.activation_matrices));

        // Serializing weight matrices
        buffer.extend(self.serialize_vec(&ann.weight_matrices));

        // Serializing bias matrices
        buffer.extend(self.serialize_vec(&ann.bias_matrices));

        Ok(buffer)
    }

    fn deserialize_header(cursor: &mut Cursor<&Vec<u8>>) -> Result<(), io::Error> {
        let mut header_magic = vec![0; 10];
        cursor.read_exact(&mut header_magic)?;
        if String::from_utf8(header_magic.to_vec()) != Ok("SHIRIN_ANN".to_string()) {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "File is not Shirin ANN model or is curropted",
            ));
        }
        Ok(())
    }

    fn check_remaining_bytes(
        cursor: &mut Cursor<&Vec<u8>>,
        desired_len: usize,
    ) -> Result<(), io::Error> {
        let total_length = cursor.get_ref().len();
        let current_position = cursor.position() as usize;
        if total_length - current_position < desired_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "File is not complete!",
            ));
        }
        Ok(())
    }

    fn deserialize_learning_rate(cursor: &mut Cursor<&Vec<u8>>) -> Result<f32, io::Error> {
        Self::check_remaining_bytes(cursor, 4)?;

        let mut buffer = [0u8; 4];
        cursor.read_exact(&mut buffer)?;
        let learning_rate: f32 = f32::from_le_bytes(buffer);
        Ok(learning_rate)
    }

    fn deserialize_example_number(cursor: &mut Cursor<&Vec<u8>>) -> Result<usize, io::Error> {
        Self::check_remaining_bytes(cursor, 4)?;

        let mut buffer = [0u8; 4];
        cursor.read_exact(&mut buffer)?;
        let example_number: usize = u32::from_le_bytes(buffer) as usize;
        Ok(example_number)
    }

    fn deserialize_layers(&self, cursor: &mut Cursor<&Vec<u8>>) -> Result<Vec<Layer>, io::Error> {
        let layers_item_len: usize = Self::read_bytes_as_usize(cursor)?;
        let layers_len: usize = Self::read_bytes_as_usize(cursor)?;

        Self::check_remaining_bytes(cursor, layers_len)?;

        let mut buffer = vec![0u8; layers_len];
        cursor.read_exact(&mut buffer)?;

        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..layers_item_len {
            let size: usize =
                u32::from_le_bytes(buffer[i * 5..i * 5 + 4].try_into().unwrap()) as usize;
            let activation_func_number =
                u8::from_le_bytes(buffer[i * 5 + 4..i * 5 + 5].try_into().unwrap());
            let activation_function = ActivationFunction::try_from(activation_func_number)
                .map_err(|_err| {
                    io::Error::new(
                        ErrorKind::InvalidData,
                        "Activation function selected is not valid",
                    )
                })?;
            let layer = Layer {
                size,
                activation_function,
            };

            layers.push(layer);
        }

        Ok(layers)
    }

    fn read_bytes_as_usize(cursor: &mut Cursor<&Vec<u8>>) -> std::io::Result<usize> {
        let mut buffer = [0u8; 4];
        cursor.read_exact(&mut buffer)?;
        Ok(u32::from_le_bytes(buffer) as usize)
    }

    fn deserialize_vec(cursor: &mut Cursor<&Vec<u8>>) -> Result<Vec<Array2<f32>>, io::Error> {
        let vec_item_len: usize = Self::read_bytes_as_usize(cursor)?;
        println!("vec_item_len {}", vec_item_len);
        let vec_len: usize = Self::read_bytes_as_usize(cursor)?;
        println!("vec_len {}", vec_len);

        Self::check_remaining_bytes(cursor, vec_len)?;

        let mut vec: Vec<Array2<f32>> = Vec::new();
        for _ in 0..vec_item_len {
            let pos = cursor.position();
            cursor.set_position(pos);
            let buffer_len: usize = Self::read_bytes_as_usize(cursor)?;
            const F32_SIZE: usize = std::mem::size_of::<f32>();
            if (vec_len % F32_SIZE) != 0 {
                return Err(io::Error::new(ErrorKind::InvalidData, "File is corrupted!"));
            }

            let cols: usize = Self::read_bytes_as_usize(cursor)?;
            let rows: usize = Self::read_bytes_as_usize(cursor)?;
            let data_buffer_len: usize = buffer_len - 2 * F32_SIZE;
            let mut buffer = vec![0u8; data_buffer_len];
            cursor.read_exact(&mut buffer)?;
            let mut array_vec: Vec<f32> = Vec::new();
            for j in 0..cols * rows {
                let value = f32::from_le_bytes(
                    buffer[j * F32_SIZE..j * F32_SIZE + F32_SIZE]
                        .try_into()
                        .unwrap(),
                );
                array_vec.push(value);
            }

            let array: Array2<f32> =
                Array2::from_shape_vec((cols, rows), array_vec).map_err(|_e| {
                    return io::Error::new(ErrorKind::InvalidInput, "Could not create 2D Array!");
                })?;
            vec.push(array);
        }
        Ok(vec)
    }

    fn deserialize_cost_function(cursor: &mut Cursor<&Vec<u8>>) -> Result<CostFunction, io::Error>{
        let mut byte: [u8; 1] = [0u8; 1];
        cursor.read_exact(&mut byte)?;
        let cost_byte = u8::from_le_bytes(byte);
        let cost_function = CostFunction::try_from(cost_byte)
            .map_err(|_err| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    "Activation function selected is not valid",
                )
            })?;

        Ok(cost_function)
    }

    pub fn deserialize(&self, buffer: &Vec<u8>) -> Result<ANN, io::Error> {
        let mut cursor = Cursor::new(buffer);

        Self::deserialize_header(&mut cursor)?;

        let layers = self.deserialize_layers(&mut cursor)?;

        let learning_rate = Self::deserialize_learning_rate(&mut cursor)?;

        let example_number = Self::deserialize_example_number(&mut cursor)?;

        let cost_function = Self::deserialize_cost_function(&mut cursor)?;

        let logit_matrices = Self::deserialize_vec(&mut cursor)?;

        let activation_matrices = Self::deserialize_vec(&mut cursor)?;

        let weight_matrices = Self::deserialize_vec(&mut cursor)?;

        let bias_matrices = Self::deserialize_vec(&mut cursor)?;

        let ann = ANN {
            learning_rate,
            example_number,
            logit_matrices,
            activation_matrices,
            weight_matrices,
            bias_matrices,
            layers,
            cost_function
        };

        Ok(ann)
    }
}
