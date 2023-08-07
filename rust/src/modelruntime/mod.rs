pub mod jitmodelruntime;
pub mod onnxmodelruntime;
pub mod dummymodelruntime;

use serde::{Deserialize, Serialize};


pub trait ModelEngine {
    fn recommend(&self, session_items: &Vec<i64>) -> Vec<i64>;
    fn get_model_device_name(&self) -> String;
    fn get_model_qty_threads(&self) -> i32;
    fn get_model_filename(&self) -> String;
}


#[allow(non_snake_case)]
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelPayload {
    max_seq_length: u32,
    C: u32,
    idx2item: Vec<u64>,
}

