use ndarray::Array2;

use crate::ann::ANN;

use super::Layer;
pub struct Serializer{}

impl Serializer {
    fn serialize_layers(&self, vec: &Vec<Layer>) -> Vec<u8>{
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
            for a in act_mat {
                act_vec.extend(a.to_le_bytes());
            }
        }
        buffer.extend((act_vec.len() as u32).to_le_bytes());
        buffer.extend(act_vec);
        buffer
    }

    pub fn serialize(&self, ann: &ANN) -> Result<Vec<u8>, std::io::Error>{
        let mut buffer: Vec<u8> = Vec::new();

        // Adding Header
        let header = "SHIRIN_ANN".as_bytes();
        buffer.extend(header);

        // Serializing layers
        buffer.extend(self.serialize_layers(&ann.layers));

        // Serializing learning_rate
        buffer.extend(ann.learning_rate.clone().to_le_bytes());

        // Serializing example_number
        buffer.extend(ann.example_number.clone().to_le_bytes());

        // Serializing activation_matrices
        buffer.extend(self.serialize_vec(&ann.activation_matrices));

        // Serializing weight matrices
        buffer.extend(self.serialize_vec(&ann.weight_matrices));

        // Serializing bias matrices
        buffer.extend(self.serialize_vec(&ann.bias_matrices));


        Ok(buffer)
    }
}
