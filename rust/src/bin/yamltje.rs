
use serde::{Deserialize, Serialize};
use serde_yaml::{self};

#[derive(Debug, Serialize, Deserialize)]
struct ModelPayload {
    max_seq_length: u32,
    C: u32,
    idx2item: Vec<u64>,
}

fn main() {
    let path = std::env::args().nth(1).expect("Please provide path to payload.yaml");
    let f = std::fs::File::open(path).expect("Could not open file.");
    let scrape_config : ModelPayload = serde_yaml::from_reader(f).expect("Could not read values.");

    println!("{:?}", scrape_config);

}