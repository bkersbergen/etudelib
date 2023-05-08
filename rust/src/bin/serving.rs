extern crate tch;

use std::sync::Arc;
use tch::{Device, Kind, TchError, Tensor};
use actix_web::middleware::{Compress, Logger};
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder, middleware, web, HttpRequest};
use actix_web::{web::{
    Data,
    Json,
}};
use actix_web::http::header;
use actix_web::http::header::ContentEncoding;
use actix_web::web::Query;
use serde::{Deserialize, Serialize};
use serde_yaml::{self};


#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelPayload {
    max_seq_length: u32,
    C: u32,
    idx2item: Vec<u64>,
}


#[derive(Debug, Deserialize, Serialize)]
pub struct V1RequestParams {
    item_ids: Vec<i64>,
    session_id: String,
}


#[post("/v1/recommend")]
async fn v1_recommend(
    app_data: Data<AppData>,
    model_payload: Data<ModelPayload>,
    query: Json<V1RequestParams>,
) -> impl Responder {
    let session_items: Vec<i64> = query.item_ids.clone();
    let device = app_data.device.as_ref();
    let model = app_data.model.as_ref();

    let max_seq_length = model_payload.max_seq_length.clone() as usize;

    println!("{:?}", session_items);
    let kind = Kind::Int64;

    let item_seq_len = session_items.len();
    let input = if item_seq_len >= max_seq_length {
        // Create a new tensor of the desired shape with zeros as the default value.
        let mut tensor = Tensor::zeros(&[1, max_seq_length as i64], (kind, *device));
        // Copy the last `length` values of the input vector into the tensor.
        let start = if item_seq_len > max_seq_length { item_seq_len - max_seq_length } else { 0 };
        tensor.copy_(&Tensor::of_slice(&session_items[start..]));
        tensor
    } else {
        // Create a new tensor of the desired shape with zeros as the default value.
        let tensor = Tensor::zeros(&[1, max_seq_length as i64], (kind, *device));
        // Copy the input values into the first `input.len()` positions of the tensor.
        tensor.narrow(1, 0, item_seq_len as i64).copy_(&Tensor::of_slice(&session_items));
        tensor
    };
    // Apply the model to the input tensor to perform inference
    let model_result = model.forward_ts(&[input, Tensor::from(item_seq_len as i32)]).unwrap();


    // // sort the probabilities in descending order and get their index positions
    // let (_sorted_probs, sorted_indexes) = model_result.sort(-1, true);
    // let sorted_indexes: Vec<i64> = Vec::from(sorted_indexes);

    println!("{:?}", model_result);
    let vec:Vec<i64> = Vec::from(model_result);
    HttpResponse::Ok().json(vec)
}


#[derive(Serialize)]
pub struct GenericResponse {
    pub status: String,
    pub message: String,
}

#[get("/ping")]
async fn ping() -> impl Responder {
    const MESSAGE: &str = "Actix Web is running";

    let response_json = &GenericResponse {
        status: "success".to_string(),
        message: MESSAGE.to_string(),
    };

    HttpResponse::Ok().json(response_json)
}

#[get("/")]
async fn home() -> impl Responder {
    HttpResponse::Found()
        .header(header::LOCATION, "/ping")
        .finish()
}

pub struct AppData {
    pub model: Arc<tch::CModule>,
    pub device: Arc<tch::Device>
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "debug");
    }
    env_logger::init();

    println!("Actix Server started successfully");
    let model_path = std::env::args().nth(1).expect("no path to a model.pt given");
    let payload_path = std::env::args().nth(2).expect("np path to payload.yaml given");

    let f = std::fs::File::open(payload_path).expect("Could not open payload file.");
    let model_payload : ModelPayload = serde_yaml::from_reader(f).expect("Could not read values.");


    let device = Device::cuda_if_available();


    // Load the model from the saved JIT script
    let model = Arc::new(tch::CModule::load_on_device(model_path, device).unwrap());
    let device = Arc::new(device);
    HttpServer::new(move || {
        let app_data = AppData {
            model: model.clone(),
            device: device.clone(),
        };
        let model_payload = model_payload.clone();
        App::new()
            .service(home)
            .service(ping)
            .service(v1_recommend)
            .app_data(Data::new(app_data))
            .app_data(Data::new(model_payload))
            .wrap(Logger::default())
            .wrap(
                middleware::DefaultHeaders::new()
                    .header("Cache-Control", "no-cache, no-store, must-revalidate")
                    .header("Pragma", "no-cache")
                    .header("Expires", "0"),
            )
    })
        .bind(("127.0.0.1", 7080)).unwrap_or_else(|_| panic!("Could not bind server to address"))
        .run()
        .await
}
