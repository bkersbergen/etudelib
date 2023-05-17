use crate::modelruntime::{ModelEngine, ModelPayload};

use ndarray::{Array, ViewRepr, IxDyn, ArrayBase};
use ort::{tensor::{DynOrtTensor, InputTensor, OrtOwnedTensor}, Environment, ExecutionProvider, SessionBuilder, LoggingLevel, Session, GraphOptimizationLevel};
use core::option::Option;
use tch::Device;


pub struct OnnxModelRuntime {
    payload: ModelPayload,
    session: Session,
}


impl OnnxModelRuntime {
    pub fn new(model_path: &String, payload_path: &String, qty_model_threads: &usize) -> OnnxModelRuntime {
        println!("number of onnx threads: {qty_model_threads}");

        let payload_file = std::fs::File::open(payload_path).expect("Could not open payload file.");
        let payload: ModelPayload = serde_yaml::from_reader(payload_file).expect("Could not read values.");

        let environment = Environment::builder()
            .with_log_level(LoggingLevel::Info)
            .build().unwrap().into_arc();

        let session = if Device::cuda_if_available().is_cuda() {
            print!("Onnx with CUDA and CPU provider");
            SessionBuilder::new(&environment).expect("unable to create a session")
                .with_intra_threads(*qty_model_threads as i16).unwrap()
                .with_inter_threads(*qty_model_threads as i16).unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
                .with_execution_providers([ExecutionProvider::cuda(), ExecutionProvider::cpu()]).unwrap()
                .with_model_from_file(model_path)
                .unwrap()
        } else {
            print!("Onnx with CPU provider");
            SessionBuilder::new(&environment).expect("unable to create a session")
                .with_intra_threads(*qty_model_threads as i16).unwrap()
                .with_inter_threads(*qty_model_threads as i16).unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
                .with_model_from_file(model_path)
                .unwrap()
        };


        OnnxModelRuntime {
            payload,
            session,
        }
    }
}

impl ModelEngine for OnnxModelRuntime {
    fn recommend(&self, session_items: &Vec<i64>) -> Vec<i64>{
        let max_seq_length = self.payload.max_seq_length as usize;
        let item_seq_len = session_items.len();

        // Variables to hold any input tensors
        let mut item_id_tensor: Option<InputTensor> = None;
        let mut mask_tensor: Option<InputTensor> = None;

        for input in self.session.inputs.iter() {
            if input.name == "item_id_list" {
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
                item_id_tensor = Some(InputTensor::Int64Tensor(Array::from_shape_vec((1, max_seq_length), zero_vec).unwrap().into_dyn()));
            } else if input.name == "max_seq_length" {
                mask_tensor = Some(InputTensor::Int64Tensor(Array::from_shape_vec(1, vec![item_seq_len as i64]).unwrap().into_dyn()));
            }
        }
        let combined_tensor: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = match (item_id_tensor, mask_tensor) {
            (Some(item_id), None) => {
                self.session.run([item_id]).unwrap()
            }
            (Some(item_id), Some(mask)) => {
                self.session.run([item_id, mask]).unwrap()
            }
            _ => {
                self.session.run([]).unwrap()
                // None None if both tensors are missing
            },
        };

        let output =combined_tensor.get(0).unwrap();
        // Usage example
        let values: OrtOwnedTensor<'_, i64, ndarray::Dim<ndarray::IxDynImpl>>  = output.try_extract().unwrap();

        let x: &ArrayBase<ViewRepr<&i64>, IxDyn> = &values.view().clone().into_dyn();
        let y: Vec<i64> = x.iter().copied().collect();
        y
    }
}



#[cfg(test)]
mod onnxmodelruntime_test {
    use super::*;

    #[test]
    fn should_happyflow_onnxmodel() {
        let mut undertest = OnnxModelRuntime::new("../../model_store/sasrec_bolcom_c10000_t50_onnx.pth".parse().unwrap(), "../../model_store/sasrec_bolcom_c10000_t50_payload.yaml".parse().unwrap(), );
        let session_items: Vec<i64> = vec![1, 5, 7, 1];
        let actual = undertest.recommend(&session_items);
        assert_eq!(actual.len(), 21);
    }


}