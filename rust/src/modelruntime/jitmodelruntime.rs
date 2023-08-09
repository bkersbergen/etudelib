use std::path::Path;
use tch::{CModule, Device, Tensor};
use crate::modelruntime::{ModelEngine, ModelPayload};
use serde_yaml::{self};

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
        let model = tch::CModule::load_on_device(model_path, device).unwrap();

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