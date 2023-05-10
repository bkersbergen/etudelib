use ndarray::{Array, ViewRepr, IxDyn, ArrayBase};
use ort::{tensor::{DynOrtTensor, InputTensor, OrtOwnedTensor}, Environment, SessionBuilder, LoggingLevel};
use core::option::Option;
use std::time::Instant;
use serving::modelruntime::jitmodelruntime::JITModelRuntime;
use serving::modelruntime::ModelEngine;

fn main() {
    let model_path = std::env::args().nth(1).expect("no path to a model.onnx given");
    println!("loading model: {model_path:?}");

    let payload_path = std::env::args().nth(2).expect("no path to a payload.yaml given");
    println!("loading payload meta data: {payload_path:?}");

    let mut undertest = JITModelRuntime::new(&model_path, &payload_path);
    let session_items: Vec<i64> = vec![1, 5, 7, 1];

    for _warmup in 0..100 {
        undertest.recommend(&session_items);
    }
    let n = 1000;
    let start = Instant::now();
    for _ in 0..n {
        let actual = undertest.recommend(&session_items);
    }
    let duration = start.elapsed();
    println!("Time elapsed in expensive_function() is: {:?}", duration);
    println!("Avg prediction took: {:?}", duration / n);

}