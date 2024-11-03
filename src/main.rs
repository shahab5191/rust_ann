mod ann;
use std::time::Instant;

use ann::ANN;
fn main() {
    let mut ann = ANN::new(vec![160000, 1000, 800, 900, 640, 500, 800, 1000], 0.01, 1);
    ann.initialize_parameters();
    let start = Instant::now();
    ann.forward_propagation();
    let duration = start.elapsed();
    println!("Elapsed time: {:?}", duration);
}
