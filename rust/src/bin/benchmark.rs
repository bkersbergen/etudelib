use ndarray::{Array, ViewRepr, IxDyn, ArrayBase};
use ort::{tensor::{DynOrtTensor, InputTensor, OrtOwnedTensor}, Environment, SessionBuilder, LoggingLevel};
use core::option::Option;
use std::time::Instant;
use serving::modelruntime::jitmodelruntime::JITModelRuntime;
use serving::modelruntime::ModelEngine;
use serving::modelruntime::onnxmodelruntime::OnnxModelRuntime;

fn main() {
    let qty_logical_cores = num_cpus::get();
    let jit_model_path = std::env::args().nth(1).expect("no path to a model.jitopt given");
    println!("loading model: {jit_model_path:?}");

    let onnx_model_path = std::env::args().nth(2).expect("no path to a model.onnx given");
    println!("loading model: {onnx_model_path:?}");

    let payload_path = std::env::args().nth(3).expect("no path to a payload.yaml given");
    println!("loading payload meta data: {payload_path:?}");

    let undertest = JITModelRuntime::new(&jit_model_path, &payload_path, &qty_logical_cores);
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
    println!("Time elapsed JIT is: {:?}", duration);
    println!("Avg prediction took: {:?}", duration / n);


    let undertest = OnnxModelRuntime::new(&onnx_model_path, &payload_path, &qty_logical_cores);
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
    println!("Time elapsed ONNX is: {:?}", duration);
    println!("Avg prediction took: {:?}", duration / n);

}