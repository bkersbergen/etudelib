use std::path::Path;
use tch::{CModule, Device, Tensor};
use crate::modelruntime::{Batch, Config, ModelEngine, ModelInput, ModelOutput, ModelPayload};
use serde_yaml::{self};
use std::{env, fs};
use std::fmt::{Debug, Formatter};
use tch::nn::{Linear, Module, Sequential};

pub struct JITModelRuntime {
    payload: ModelPayload,
    model: CModule,
    device: Device,
    device_name: String,
    qty_model_threads: i32,
    model_filename: String,
}

impl JITModelRuntime {
    pub fn new(model_path: &String, payload_path: &String, qty_threads: &usize) -> JITModelRuntime {
        println!("Number of jit threads: {qty_threads}");

        println!("Cuda available: {}", tch::Cuda::is_available());
        println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
        tch::set_num_threads(*qty_threads as i32);
        tch::set_num_interop_threads(*qty_threads as i32);
        let device = Device::cuda_if_available();
        let payload_file = std::fs::File::open(payload_path).expect("Could not open payload file.");
        let payload: ModelPayload = serde_yaml::from_reader(payload_file).expect("Could not read values.");
        let device_name = if device.is_cuda() { "cuda".to_string() } else { "cpu".to_string() };
        let qty_model_threads = *qty_threads as i32;
        let model_filename = Path::new(model_path)
            .file_name()
            .and_then(|os_str| os_str.to_str())
            .unwrap_or_default().to_string();

        if device.is_cuda() {
            println!("JIT using CUDA and CPU");
            println!("version_cudnn: {}", tch::utils::version_cudnn());
            println!("version_cudart: {}", tch::utils::version_cudart());
        } else {
            println!("JIT using only CPU");
        }
        let model = tch::CModule::load_on_device(model_path, Device::Cpu).unwrap();

        JITModelRuntime {
            payload,
            model,
            device,
            device_name: device_name,
            qty_model_threads: qty_model_threads,
            model_filename,
        }
    }
}

impl ModelEngine for JITModelRuntime {
    fn recommend_batch(&self, batched_input: Batch<ModelInput>) -> Batch<ModelOutput> {
        let max_seq_length = self.payload.max_seq_length as usize;
        let batch_size = batched_input.len();
        let mut padded_batch: Vec<i64> = Vec::with_capacity(batch_size * max_seq_length);
        let mut mask: Vec<i32> = Vec::with_capacity(batch_size);

        // Pad and flatten the batched input
        for session_items in &batched_input {
            let item_seq_len = session_items.len();

            let mut zero_vec: Vec<i64> = vec![0; max_seq_length];
            let start_index = if item_seq_len > max_seq_length {
                item_seq_len - max_seq_length
            } else {
                0
            };

            for (i, &value) in session_items[start_index..].iter().enumerate() {
                zero_vec[i] = value;
            }
            for &value in &zero_vec {
                padded_batch.push(value);
            }
            mask.push(item_seq_len as i32);
        }

        // Convert the padded batch into a Tensor
        let input = Tensor::of_slice(&padded_batch)
            .view((batch_size as i64, -1)) // Reshape to have batch size of the actual batch size
            .to_device(self.device);

        // Create the mask for the input items on the same device as the model
        let input_mask = Tensor::of_slice(&mask)
            .to_device(self.device);

        // Perform batch prediction in one call
        let result_tensor = self.model.forward_ts(&[input, input_mask]).unwrap();
        Vec::from(result_tensor.to_device(Device::Cpu))
    }

    fn recommend(&self, session_items: &Vec<i64>) -> Vec<i64>{
        let max_seq_length = self.payload.max_seq_length as usize;
        let item_seq_len = session_items.len();

        // Create padding input vector from the session_items
        let mut zero_vec: Vec<i64> = vec![0; max_seq_length];
        let start_index = if item_seq_len > max_seq_length {
            item_seq_len - max_seq_length
        } else {
            0
        };

        for (i, &value) in session_items[start_index..].iter().enumerate() {
            zero_vec[i] = value;
        }
        // Convert to a Tensor and put in on the same device as the model
        let input = Tensor::of_slice(&zero_vec)
            .view((1, -1)) // Reshape to have batch size of 1
            .to_device(self.device)
            ;
        // Create the mask for the input items on the same device as the model
        let input_mask = Tensor::of_slice(&[item_seq_len as i32]).to_device(self.device);
        let result_tensor = self.model.forward_ts(&[input, input_mask]).unwrap();
        Vec::from(result_tensor.to_device(Device::Cpu))
    }

    fn get_model_device_name(&self) -> String {
        self.device_name.clone()
    }

    fn get_model_qty_threads(&self) -> i32 {
        self.qty_model_threads.clone()
    }

    fn get_model_filename(&self) -> String {
        self.model_filename.clone()
    }

    fn load() -> Self {
        println!("lazy loading model and config");
        // Get the current working directory
        let current_dir = std::env::current_dir().expect("Failed to get current directory");
        let serving_yml_path = current_dir.join("config").join("serving.yaml");
        println!("opening {:?}", &serving_yml_path);
        let f = std::fs::File::open(serving_yml_path).expect("Could not open configfile.");
        let config: Config = serde_yaml::from_reader(f).expect("Could not read values.");
        println!("{:?}", config);

        let qty_logical_cores = num_cpus::get();
        let qty_physical_cores = num_cpus::get_physical();
        println!("AAANumber of logical cores: {qty_logical_cores}");
        println!("AAANumber of physical cores: {qty_physical_cores}");

        let device = Device::cuda_if_available();
        let qty_threads = 1;

        // Attempt to open the 'serving.yml' file
        let f = std::fs::File::open(config.payload_path).expect("Could not open configfile.");
        let payload: ModelPayload = serde_yaml::from_reader(f).expect("Could not read configfile values.");
        let device_name = if device.is_cuda() { "cuda".to_string() } else { "cpu".to_string() };
        let qty_model_threads = qty_threads;

        if device.is_cuda() {
            println!("AAAJIT using CUDA and CPU");
            println!("version_cudnn: {}", tch::utils::version_cudnn());
            println!("version_cudart: {}", tch::utils::version_cudart());
        } else {
            println!("AAAJIT using only CPU");
        }

        let path = Path::new(&config.model_path);

        // Check if the file exists before attempting to load it.
        if !path.exists() {
            println!("File not found: {}", &config.model_path);
        } else {
            println!("File exists: {}", &config.model_path);
        }
        let model = tch::CModule::load_on_device(&config.model_path, device).unwrap();

        Self {
            payload,
            model,
            device,
            device_name: device_name,
            qty_model_threads: qty_model_threads,
            model_filename: config.model_path,
        }
    }
}



#[cfg(test)]
mod jitmodelruntime_test {
    use super::*;

    #[test]
    fn should_happyflow_jitmodel() {
        let undertest = JITModelRuntime::new(&"/Users/bkersbergen/phd/etudelib/rust/model_store/noop_bolcom_c10000_t50_jitopt.pth".to_string(), &"/Users/bkersbergen/phd/etudelib/rust/model_store/noop_bolcom_c10000_t50_payload.yaml".to_string(), &1);
        let session_items: Vec<i64> = vec![1, 5, 7, 1];
        let actual = undertest.recommend(&session_items);
        assert_eq!(actual.len(), 21);
    }


}