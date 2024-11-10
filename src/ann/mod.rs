mod serializer;

use std::{f32::consts::E, fs, io, path::PathBuf, usize, vec};

use log::info;
use ndarray::{Array, Array2, Axis};
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

trait Pow {
    fn pow(&self, val: i32) -> Array2<f32>;
}

impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.map(|x| x.log(std::f32::consts::E))
    }
}

impl Pow for Array2<f32> {
    fn pow(&self, val: i32) -> Array2<f32> {
        self.map(|x| x.powi(val))
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

impl TryFrom<u8> for CostFunction {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(CostFunction::MeanSquaredError),
            2 => Ok(CostFunction::BinaryCrossEntropy),
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
    pub cost_function: CostFunction,
    pub logit_matrices: Vec<Array2<f32>>,
    pub activation_matrices: Vec<Array2<f32>>,
    pub weight_matrices: Vec<Array2<f32>>,
    pub bias_matrices: Vec<Array2<f32>>,
}

impl ANN {
    pub fn new(
        layers: Vec<Layer>,
        learning_rate: f32,
        example_number: usize,
        cost_function: CostFunction,
    ) -> Self {
        let mut activation_matrices: Vec<Array2<f32>> = vec![];
        let mut weight_matrices: Vec<Array2<f32>> = vec![];
        let mut bias_matrices: Vec<Array2<f32>> = vec![];
        let mut logit_matrices: Vec<Array2<f32>> = vec![];
        activation_matrices.push(Array2::<f32>::ones((layers[0].size, example_number)));

        for i in 1..layers.len() {
            let act_mat = Array2::<f32>::zeros((layers[i].size, example_number));
            let bias_mat = Array2::<f32>::zeros((layers[i].size, 1));
            let weight_mat = Array2::<f32>::zeros((layers[i].size, layers[i - 1].size));
            let logit_mat = Array2::<f32>::zeros((layers[i].size, example_number));
            activation_matrices.push(act_mat);
            bias_matrices.push(bias_mat);
            weight_matrices.push(weight_mat);
            logit_matrices.push(logit_mat);
        }

        ANN {
            layers,
            learning_rate,
            example_number,
            logit_matrices,
            activation_matrices,
            weight_matrices,
            bias_matrices,
            cost_function,
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
    fn sigmoid(num: &f32) -> f32 {
        1.0 / (1.0 + E.powf(-num))
    }

    fn sigmoid_activation(arr: &Array2<f32>) -> Array2<f32> {
        info!("Calculating sigmoid activation function");
        arr.map(Self::sigmoid)
    }

    fn sigmoid_backward(&self, layer_number: usize) -> Array2<f32> {
        info!("Calculating sigmoid for layer {layer_number}");
        fn sigmoid_prime(val: &f32) -> f32 {
            ANN::sigmoid(val) * (1.0 - ANN::sigmoid(val))
        }

        self.logit_matrices[layer_number].map(|x| sigmoid_prime(x))
    }

    fn relu_activation(arr: &Array2<f32>) -> Array2<f32> {
        info!("Calculating relu activation function");
        fn relu(num: &f32) -> f32 {
            match *num > 0.0 {
                true => *num,
                false => 0.0,
            }
        }
        arr.map(relu)
    }

    fn relu_backward(&self, layer_number: usize) -> Array2<f32> {
        info!("Calculating relu for layer {layer_number}");
        fn relu_prime(val: &f32) -> f32 {
            match *val > 0.0 {
                true => 1.0,
                false => 0.0,
            }
        }
        self.logit_matrices[layer_number].map(|x| relu_prime(x))
    }

    // TODO: Finish this
    fn compute_delta(
        &self,
        next_delta: &Array2<f32>,
        layer_number: usize,
    ) -> Result<Array2<f32>, io::Error> {
        info!("Generating delta for layer {}", layer_number);
        let activation_derivative = match self.layers[layer_number].activation_function {
            ActivationFunction::Relu => self.relu_backward(layer_number),
            ActivationFunction::Sigmoid => self.sigmoid_backward(layer_number),
        };
        info!("Delta activation: {:?}", activation_derivative);

        let delta_batch =
            (activation_derivative * &self.weight_matrices[layer_number]).t().dot(next_delta);

        let delta = delta_batch.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));

        Ok(delta)
    }

    fn compute_gradients(
        delta: &Array2<f32>,
        activation_matrix: &Array2<f32>,
    ) -> Result<Array2<f32>, io::Error> {
        info!("Computing weight gradients");
        Ok(delta.dot(&activation_matrix.t()))
    }

    pub fn backpropagation(
        &self,
        expected: &Array2<f32>,
    ) -> Result<(f32, Vec<Array2<f32>>, Vec<Array2<f32>>), io::Error> {
        info!("Back propagation started");
        let cost = self.cost(expected)?;
        info!("Cost result: {cost}");
        let layers_len = self.layers.len();

        let mut grads: Vec<Array2<f32>> = Vec::new();
        let mut deltas: Vec<Array2<f32>> = Vec::new();

        for i in 1..self.layers.len() {
            grads.push(Array2::zeros((self.layers[i].size, 1)));
            deltas.push(Array2::zeros((self.layers[i].size, 1)));
        }
        info!("Grads and deltas initialized");

        let delta_cost = match self.cost_function {
            CostFunction::MeanSquaredError => self.mean_squared_error_prime(expected),
            CostFunction::BinaryCrossEntropy => self.binary_cross_entropy_prime(expected),
        }
        .mean_axis(Axis(1))
        .unwrap()
        .insert_axis(Axis(1));

        info!(
            "delta cost in respect to output generated:\n{:?}",
            delta_cost
        );

        let output_activation_derivative = match self.layers.last().unwrap().activation_function {
            ActivationFunction::Relu => self.relu_backward(layers_len - 2),
            ActivationFunction::Sigmoid => self.sigmoid_backward(layers_len - 2),
        };

        info!(
            "output activation derivation: {:?}",
            output_activation_derivative
        );

        let delta_output = output_activation_derivative * delta_cost;

        info!("output delta: {:?}", delta_output);

        deltas.push(delta_output);
        info!("output grad and output delta added to vector!");

        for i in (0..layers_len - 1).rev() {
            deltas[i] = self.compute_delta(&deltas[i], i)?;
        }

        for i in 0..layers_len - 1 {
            grads[i] = Self::compute_gradients(&deltas[i + 1], &self.activation_matrices[i])?;
        }

        Ok((cost, deltas, grads))
    }

    pub fn train(&mut self, expected: &Array2<f32>) -> Result<f32, io::Error> {
        info!("Training model...");
        let (cost, deltas, grads) = self.backpropagation(expected)?;


        info!("Changing weights and biases based on generated grades and deltas");
        for i in 0..self.weight_matrices.len() {
            self.weight_matrices[i] = &self.weight_matrices[i] - self.learning_rate * &grads[i];
            self.bias_matrices[i] = &self.bias_matrices[i] - self.learning_rate * &deltas[i + 1];
        }

        Ok(cost)
    }

    fn linear_forward_activation(
        &mut self,
        layer_number: usize,
        activation: ActivationFunction,
    ) -> () {
        info!(
            "Linear forward activation function called for layer {}",
            layer_number
        );
        let weight_mat = self.weight_matrices[layer_number].clone();
        let act_mat = self.activation_matrices[layer_number].clone();
        let bias_mat = self.bias_matrices[layer_number].clone();

        info!("weight shape: {:?}, activation shape: {:?}, bias shape: {:?}", weight_mat.shape(), act_mat.shape(), bias_mat.shape());

        let logit_mat = weight_mat.dot(&act_mat) + &bias_mat;
        let next_activation_mat = match activation {
            ActivationFunction::Relu => Self::relu_activation(&logit_mat),
            ActivationFunction::Sigmoid => Self::sigmoid_activation(&logit_mat),
        };
        self.logit_matrices[layer_number] = logit_mat;
        self.activation_matrices[layer_number + 1] = next_activation_mat;
    }

    pub fn forward_propagation(&mut self) {
        info!("Forward propagation called");
        info!("Length of layers is {}", self.layers.len());
        for l in 0..self.layers.len() - 1 {
            self.linear_forward_activation(l, self.layers[l].activation_function.clone());
        }
    }

    fn is_equal_size<T>(arr1: &Array2<T>, arr2: &Array2<T>) -> Result<(), io::Error> {
        info!("Checking if two arrays have equal dimensions");
        info!("Array1: {:?}, Array2: {:?}", arr1.dim(), arr2.dim());
        if arr1.dim() != arr2.dim() {
            info!("Arrays dimensions are not equal!");
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Provided result do not have same size as model output",
            ));
        }
        info!("Arrays dimensionsa ARE equal!");

        Ok(())
    }

    fn binary_cross_entropy(
        output: &Array2<f32>,
        expected: &Array2<f32>,
    ) -> Result<Array2<f32>, io::Error> {
        info!("Binary cross entropy called");
        Self::is_equal_size(output, expected)?;
        let example_number = output.dim().1;
        let cost = -(1.0 / example_number as f32)
            * (expected.dot(&(output + 1e-8).log()) + (1.0 - expected) * (1.0 - output).log());

        Ok(cost)
    }

    fn binary_cross_entropy_prime(&self, expected: &Array2<f32>) -> Array2<f32> {
        let output = self.activation_matrices.last().unwrap();
        (output - expected) / (output * (1.0 - output))
    }

    fn mean_squared_error(
        output: &Array2<f32>,
        expected: &Array2<f32>,
    ) -> Result<Array2<f32>, io::Error> {
        info!("Mean squared error called");
        Self::is_equal_size::<f32>(output, expected)?;
        let example_number: f32 = output.dim().1 as f32;
        let cost = (1.0 / example_number) * (expected - output).pow(2);
        info!("Calculated cost is: {}", cost);
        Ok(cost)
    }

    fn mean_squared_error_prime(&self, expected: &Array2<f32>) -> Array2<f32> {
        2.0 * (self.activation_matrices.last().unwrap() - expected)
    }

    fn cost(&self, expected: &Array2<f32>) -> Result<f32, io::Error> {
        let output = match self.activation_matrices.last() {
            Some(val) => val,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "Model is not created!",
                ))
            }
        };

        let cost = match self.cost_function {
            CostFunction::MeanSquaredError => Self::mean_squared_error(output, expected),
            CostFunction::BinaryCrossEntropy => Self::binary_cross_entropy(&output, expected),
        };

        Ok(cost?.sum())
    }

    pub fn print_layers(&self) -> () {
        println!("Learning Rate {}", self.learning_rate);
        println!("Example Number: {}", self.example_number);
        println!("Cost Function: {:?}", self.cost_function);
        for l in 0..self.layers.len() - 1 {
            println!("Layer({}):", l);
            println!("Layer Size: {}", self.layers[l].size);
            println!("Layer Activation Function: {:?}", self.layers[l].activation_function);
            for n in &self.activation_matrices[l] {
                print!("A: {}\t", *n);
            }
            println!();
            for w in &self.weight_matrices[l] {
                print!("W: {}\t", w)
            }
            println!();
            for b in &self.bias_matrices[l] {
                print!("B: {}\t", b);
            }
            println!();
            for z in &self.logit_matrices[l] {
                print!("Z: {}\t", z);
            }
            println!();
        }
        for n in self.activation_matrices.last().unwrap() {
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
