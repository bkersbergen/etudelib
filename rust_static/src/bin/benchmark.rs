use core::option::Option;
use std::time::Instant;
use serving::modelruntime::dummyruntime::DummyRuntime;
use serving::modelruntime::ModelEngine;

fn main() {
    let qty_logical_cores = num_cpus::get();

    let undertest = DummyRuntime::new();
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
    println!("Time elapsed Dummy is: {:?}", duration);
    println!("Avg prediction took: {:?}", duration / n);
}