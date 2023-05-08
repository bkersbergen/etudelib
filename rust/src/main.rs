// extern crate tch;
// use tch::{Device, Kind, Tensor};
//
// fn main() {
//     let path = std::env::args().nth(1).expect("no path to a model.pt given");
//     let device = Device::cuda_if_available();
//     // Load the model from the saved JIT script
//     let model = tch::CModule::load_on_device(path, device).unwrap();
//
//     // Create a random input tensor to demonstrate inference
//     let input = Tensor::rand(&[1, 1, 28, 28], (Kind::Float, device));
//
//     // Apply the model to the input tensor to perform inference
//     let output = model.forward_ts(&[input]).unwrap();
//
//     println!("Output: {:?}", output);
// }