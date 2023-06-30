use crate::modelruntime::ModelEngine;

pub struct DummyRuntime {
}

impl DummyRuntime {
    pub fn new() -> DummyRuntime {
        DummyRuntime {
        }
    }
}

impl ModelEngine for DummyRuntime {
    fn recommend(&self, _session_items: &Vec<i64>) -> Vec<i64>{
        let vec: Vec<i64> = (1..=21).collect();
        vec
    }
}



#[cfg(test)]
mod dummyruntime_test {
    use super::*;

    #[test]
    fn should_happyflow_dummymodel() {
        let mut undertest = DummyRuntime::new();
        let session_items: Vec<i64> = vec![1, 5, 7, 1];
        let actual = undertest.recommend(&session_items);
        assert_eq!(actual.len(), 21);
    }


}