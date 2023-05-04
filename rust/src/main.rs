extern crate tch;
use tch::{Device, Kind, Tensor};

fn main() {
    let device = Device::cuda_if_available();
    // Load the model from the saved JIT script
    let model = tch::CModule::load_on_device("mnist_cuda.pt", device).unwrap();

    // Create a random input tensor to demonstrate inference
    let input = Tensor::rand(&[1, 1, 28, 28], (Kind::Float, Device::Cpu));

    // Apply the model to the input tensor to perform inference
    let output = model.forward_ts(&[input]).unwrap();

    println!("Output: {:?}", output);
}