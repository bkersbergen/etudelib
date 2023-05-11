// use std::path::Path;
// use std::process::id;
// use std::sync::Arc;
//
// use std::slice;
// use ndarray::{array, concatenate, s, Array1, Axis, Array, Dimension, ViewRepr, IxDyn, ArrayBase};
// use ort::{tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor}, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, LoggingLevel, AllocatorType};
// use core::option::Option;
//
// fn main() {
//     let path = std::env::args().nth(1).expect("no path to a model.onnx given");
//     println!("loading model: {path:?}");
//
//     let environment = Environment::builder()
//         .with_log_level(LoggingLevel::Info)
//         .build().unwrap().into_arc();
//
//
//     let mut session = SessionBuilder::new(&environment).expect("unable to create a session")
//         .with_model_from_file(path).unwrap();
//
//
//     let session_items = vec![1, 5, 7, 1];
//     let item_seq_len = session_items.len();
//     let max_seq_length = 50 as usize;
//     // Variables to hold any input tensors
//     let mut item_id_tensor: Option<InputTensor> = None;
//     let mut mask_tensor: Option<InputTensor> = None;
//
//     for input in session.inputs.iter() {
//         if input.name == "item_id_list" {
//             // Create padding input vector from the session_items
//             let mut zero_vec: Vec<i64> = vec![0; max_seq_length];
//             let start_index = if item_seq_len > max_seq_length {
//                 item_seq_len - max_seq_length
//             } else {
//                 0
//             };
//             for (i, &value) in session_items[start_index..].iter().enumerate() {
//                 zero_vec[i] = value;
//             }
//             item_id_tensor = Some(InputTensor::Int64Tensor(Array::from_shape_vec((1, max_seq_length), zero_vec).unwrap().into_dyn()));
//         } else if input.name == "max_seq_length" {
//             mask_tensor = Some(InputTensor::Int64Tensor(Array::from_shape_vec((1), vec![item_seq_len as i64]).unwrap().into_dyn()));
//         }
//         println!("{:?} {:?} {:?}", input.name, input.dimensions, input.input_type);
//     }
//     let combined_tensor: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = match (item_id_tensor, mask_tensor) {
//         (Some(item_id), None) => {
//             println!("inference with only session items");
//             session.run([item_id]).unwrap()
//         }
//         (Some(item_id), Some(mask)) => {
//             println!("inference with session items and mask");
//             session.run([item_id, mask]).unwrap()
//         }
//         _ => {
//             println!("inference with no input");
//             session.run([]).unwrap()
//             // None None if both tensors are missing
//         },
//     };
//
//     let output =combined_tensor.get(0).unwrap();
//     // Usage example
//     let values: OrtOwnedTensor<'_, i64, ndarray::Dim<ndarray::IxDynImpl>>  = output.try_extract().unwrap();
//
//     let x: &ArrayBase<ViewRepr<&i64>, IxDyn> = &values.view().clone().into_dyn();
//     let y: Vec<i64> = x.iter().copied().collect();
//     println!("Recommended items: {:?}", y);
// }
