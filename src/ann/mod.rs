mod serializer;

use std::{f32::consts::E, fs, io, path::PathBuf, usize, vec};

use ndarray::{Array, Array2};
use rand::{distributions::Uniform, prelude::Distribution};
use serializer::Serializer;

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    Sigmoid = 1,
    Relu = 2,
}

#[derive(Debug, Clone)]
pub enum CostFunction {
    MeanSquaredError = 1,
    BinaryCrossEntropy = 2,
}

trait Log {
    fn log(&self) -> Array2<f32>;
}

impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.map(|x| x.log(std::f32::consts::E))
    }
}

impl TryFrom<u8> for ActivationFunction {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(ActivationFunction::Sigmoid),
            2 => Ok(ActivationFunction::Relu),
            _ => Err("Invalid value for ActivationFunction"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub size: usize,
    pub activation_function: ActivationFunction,
}

#[derive(Debug, Clone)]
pub struct ANN {
    pub layers: Vec<Layer>,
    pub learning_rate: f32,
    pub example_number: usize,
    pub activation_matrices: Vec<Array2<f32>>,
    pub weight_matrices: Vec<Array2<f32>>,
    pub bias_matrices: Vec<Array2<f32>>,
}

impl ANN {
    pub fn new(layers: Vec<Layer>, learning_rate: f32, example_number: usize) -> Self {
        let mut activation_matrices: Vec<Array2<f32>> = vec![];
        let mut weight_matrices: Vec<Array2<f32>> = vec![];
        let mut bias_matrices: Vec<Array2<f32>> = vec![];
        activation_matrices.push(Array2::<f32>::ones((layers[0].size, example_number)));
        for i in 1..layers.len() {
            let act_mat = Array2::<f32>::zeros((layers[i].size, example_number));
            let bias_mat = Array2::<f32>::zeros((layers[i].size, 1));
            let weight_mat = Array2::<f32>::zeros((layers[i - 1].size, layers[i].size));
            activation_matrices.push(act_mat);
            bias_matrices.push(bias_mat);
            weight_matrices.push(weight_mat);
        }

        ANN {
            layers,
            learning_rate,
            example_number,
            activation_matrices,
            weight_matrices,
            bias_matrices,
        }
    }

    pub fn initialize_parameters(&mut self) {
        let between = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let number_of_layers = self.layers.len();

        for l in 0..number_of_layers - 1 {
            let weight_array: Vec<f32> = (0..self.layers[l + 1].size * self.layers[l].size)
                .map(|_| between.sample(&mut rng))
                .collect();

            let bias_array: Vec<f32> = (0..self.layers[l + 1].size).map(|_| 0.0).collect();

            self.weight_matrices[l] =
                Array::from_shape_vec((self.layers[l + 1].size, self.layers[l].size), weight_array)
                    .unwrap();

            self.bias_matrices[l] =
                Array2::from_shape_vec((self.layers[l + 1].size, 1), bias_array).unwrap();
        }
    }

    fn sigmoid_activation(&self, arr: Array2<f32>) -> Array2<f32> {
        fn sigmoid(num: &f32) -> f32 {
            1.0 / (1.0 + E.powf(-num))
        }

        arr.map(sigmoid)
    }

    fn relu_activation(&self, arr: Array2<f32>) -> Array2<f32> {
        fn relu(num: &f32) -> f32 {
            match *num > 0.0 {
                true => *num,
                false => 0.0,
            }
        }

        arr.map(relu)
    }

    fn linear_forward_activation(
        &mut self,
        layer_number: usize,
        activation: ActivationFunction,
    ) -> () {
        let weight_mat = self.weight_matrices[layer_number].clone();
        let act_mat = self.activation_matrices[layer_number].clone();
        let bias_mat = self.bias_matrices[layer_number].clone();

        let logit_mat = weight_mat.dot(&act_mat) + &bias_mat;
        let next_activation_mat = match activation {
            ActivationFunction::Relu => self.relu_activation(logit_mat),
            ActivationFunction::Sigmoid => self.sigmoid_activation(logit_mat),
        };

        self.activation_matrices[layer_number + 1] = next_activation_mat;
    }

    pub fn forward_propagation(&mut self) {
        for l in 0..self.layers.len() - 1 {
            self.linear_forward_activation(l, self.layers[l].activation_function.clone());
        }
    }

    pub fn cost(&self, cost_function: CostFunction, expected_res: &Array2<f32>) -> Result<f32, io::Error> {
        let output = match self.activation_matrices.last() {
            Some(val) => val,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "Model is not defined!",
                ))
            }
        };

        if expected_res.dim() != output.dim() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Provided result do not have same size as model output",
            ));
        }
        let cost = -(1.0 / self.example_number as f32)
            * (expected_res.dot(&output.clone().reversed_axes().log())
                + (1.0 - expected_res) * (1.0 - output).log()
            );
        
        Ok(cost.sum())
    }

    pub fn print_layers(&self) -> () {
        for l in 0..self.layers.len() - 1 {
            println!("Layer({}):", l);
            for n in &self.activation_matrices[l] {
                print!("A: {}\t", *n);
            }
            println!();
            for w in &self.weight_matrices[l] {
                print!("W: {}\t", w)
            }
            println!()
        }
        for n in &self.activation_matrices[self.layers.len() - 1] {
            print!("Output: {}\t", *n);
        }
        println!();
    }

    pub fn save_model(&self, file_path: PathBuf) -> Result<(), io::Error> {
        let serializer = Serializer {};
        let buffer = serializer.serialize(self)?;
        fs::write(file_path, buffer).expect("Unable to write to file!");
        Ok(())
    }

    pub fn from_file(file_path: PathBuf) -> Result<Self, io::Error> {
        let buffer = fs::read(file_path)?;

        let serializer = Serializer {};
        let ann = serializer.deserialize(&buffer)?;
        Ok(ann)
    }
}
