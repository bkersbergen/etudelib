use crate::modelruntime::ModelEngine;

pub struct DummyModelRuntime {
}

impl DummyModelRuntime {
    pub fn new() -> DummyModelRuntime {
        DummyModelRuntime {
        }
    }
}

impl ModelEngine for DummyModelRuntime {
    fn recommend(&self, _session_items: &Vec<i64>) -> Vec<i64>{
        let vec: Vec<i64> = (1..=21).collect();
        vec
    }
}



#[cfg(test)]
mod dummymodelruntime_test {
    use super::*;

    #[test]
    fn should_happyflow_dummymodel() {
        let undertest = DummyModelRuntime::new();
        let session_items: Vec<i64> = vec![1, 5, 7, 1];
        let actual = undertest.recommend(&session_items);
        assert_eq!(actual.len(), 21);
    }


}