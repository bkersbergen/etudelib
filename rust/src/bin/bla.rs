use tch::{Device, Kind, Tensor};

fn main() {

    // Define the input values as a vector.
    let session_items = vec![1, 2, 3];

    // Define the desired length of the tensor.
    let max_seq_len = 4;

    let device = Device::Cpu;
    let kind = Kind::Int64;

    let tensor = if session_items.len() >= max_seq_len {
        // Create a new tensor of the desired shape with zeros as the default value.
        let mut tensor = Tensor::zeros(&[1, max_seq_len as i64], (kind, device));
        // Copy the last `length` values of the input vector into the tensor.
        let start = if session_items.len() > max_seq_len { session_items.len() - max_seq_len } else { 0 };
        tensor.copy_(&Tensor::of_slice(&session_items[start..]));
        tensor
    } else {
        // Create a new tensor of the desired shape with zeros as the default value.
        let tensor = Tensor::zeros(&[1, max_seq_len as i64], (kind, device));
        // Copy the input values into the first `input.len()` positions of the tensor.
        tensor.narrow(1, 0, session_items.len() as i64).copy_(&Tensor::of_slice(&session_items));
        tensor
    };

    // Print the resulting tensor values.
    for i in 0..max_seq_len {
        println!("{}: {:?}", i, tensor.int64_value(&[0, i as i64]));
    }
}