use crate::modelruntime::{Batch, ModelEngine, ModelInput, ModelOutput};

pub struct DummyModelRuntime {
}

impl DummyModelRuntime {
    pub fn new() -> DummyModelRuntime {
        DummyModelRuntime {
        }
    }
}

impl ModelEngine for DummyModelRuntime {

    fn recommend_batch(&self, batched_input: Batch<ModelInput>) -> Batch<ModelOutput> {
        let mut results:Batch<ModelOutput> = Vec::with_capacity(batched_input.len());
        for (_index, value) in batched_input.iter().enumerate() {
            let result:ModelOutput = self.recommend(value);
            results.push(result);
        }
        results
    }

    fn recommend(&self, _session_items: &ModelInput) -> ModelOutput{
        // let batch_predict = batched_fn! {
        //     handler =
        // }

        let vec: ModelOutput = (1..=21).collect();
        vec
    }

    fn get_model_device_name(&self) -> String {
        "cpu".to_string()
    }

    fn get_model_qty_threads(&self) -> i32 {
        1
    }

    fn get_model_filename(&self) -> String {
        String::from("DummyModel")
    }


    fn load() -> Self {
        Self {}
    }
}



#[cfg(test)]
mod dummymodelruntime_test {
    use super::*;

    #[test]
    fn should_happyflow_dummymodel() {
        let undertest = DummyModelRuntime::new();
        let session_items: ModelInput = vec![1, 5, 7, 1];
        let actual = undertest.recommend(&session_items);
        assert_eq!(actual.len(), 21);
    }


}