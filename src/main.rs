mod ann;
use ann::{ActivationFunction, ANN};
fn main() {
    let mut ann = ANN::new(vec![10,3,2,10], 0.01, 1);
    let hash = ann.initialize_parameters();
    println!("before fa:\n{:?}",ann.activation_matrices[1]);
    ann.linear_forward_activation(0, ActivationFunction::Sigmoid);
    println!("after fa:\n{:?}",ann.activation_matrices[1]);
}
