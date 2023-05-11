pub mod jitmodelruntime;
pub mod onnxmodelruntime;

use serde::{Deserialize, Serialize};


pub trait ModelEngine {
    fn recommend(&self, session_items: &Vec<i64>) -> Vec<i64>;
}


#[allow(non_snake_case)]
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelPayload {
    max_seq_length: u32,
    C: u32,
    idx2item: Vec<u64>,
}

