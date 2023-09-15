pub mod jitmodelruntime;
pub mod onnxmodelruntime;
pub mod dummymodelruntime;

use std::net::Ipv4Addr;
use serde::{Deserialize, Serialize};

pub type ModelInput = Vec<i64>;
pub type ModelOutput = Vec<i64>;
pub type Batch<T> = Vec<T>;

pub trait ModelEngine {
    fn recommend_batch(&self, batched_input: Batch<ModelInput>) -> Batch<ModelOutput>;
    fn recommend(&self, session_items: &ModelInput) -> ModelOutput;
    fn get_model_device_name(&self) -> String;
    fn get_model_qty_threads(&self) -> i32;
    fn get_model_filename(&self) -> String;
    fn load() -> Self;

    // fn is_usable() -> Self;
}


#[allow(non_snake_case)]
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelPayload {
    max_seq_length: u32,
    C: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub host: Ipv4Addr,
    pub port: u16,
    pub qty_model_threads: usize,
    pub model_path: String,
    pub payload_path: String,
}
