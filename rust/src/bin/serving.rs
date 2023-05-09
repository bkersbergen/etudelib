extern crate tch;

use std::sync::Arc;
use tch::{Device, Kind, Tensor};
use tract_onnx::prelude::*;
use actix_web::middleware::{Logger};
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder, middleware};
use actix_web::{web::{
    Data,
    Json,
}};
use actix_web::http::header;
use serde::{Deserialize, Serialize};
use serde_yaml::{self};


#[allow(non_snake_case)]
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
    jitmodel: Data<JITModel>,
    model_payload: Data<ModelPayload>,
    query: Json<V1RequestParams>,
) -> impl Responder {
    let session_items: Vec<i64> = query.item_ids.clone();
    let device = jitmodel.device.as_ref();
    let model = jitmodel.model.as_ref();

    let max_seq_length = model_payload.max_seq_length.clone() as usize;

    let item_seq_len = session_items.len();

    // Create padding input vector from the session_items
    let mut zero_vec: Vec<i64> = vec![0; max_seq_length];
    for (i, &value) in session_items.iter().enumerate() {
        if i < zero_vec.len() {
            zero_vec[i] = value;
        } else {
            break;
        }
    }
    // Convert to a Tensor and put in on the same device as the model
    let input = Tensor::of_slice(&zero_vec)
        .view((1, -1)) // Reshape to have batch size of 1
        .to_device(*device)
        ;
    // Create the mask for the input items on the same device as the model
    let input_mask = Tensor::from(item_seq_len as i32).to_device(*device);
    let result_tensor = model.forward_ts(&[input, input_mask]).unwrap();
    let result_item_ids:Vec<i64> = Vec::from(result_tensor);

    HttpResponse::Ok().json(result_item_ids)
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

pub struct JITModel {
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

    if model_path.ends_with("_jitopt.pth") {
        println!("Loading jitopt model");
    } else if model_path.ends_with("_onnx.pth") {
        println!("Loading onnx model");
    }

    let payload_path = std::env::args().nth(2).expect("np path to payload.yaml given");
    let f = std::fs::File::open(payload_path).expect("Could not open payload file.");
    let model_payload : ModelPayload = serde_yaml::from_reader(f).expect("Could not read values.");

    let device = Device::cuda_if_available();

    // Load the model from the saved JIT script
    let model = Arc::new(tch::CModule::load_on_device(model_path, device).unwrap());
    let device = Arc::new(device);
    HttpServer::new(move || {
        let jitmodel = JITModel {
            model: model.clone(),
            device: device.clone(),
        };
        let model_payload = model_payload.clone();
        App::new()
            .service(home)
            .service(ping)
            .service(v1_recommend)
            .app_data(Data::new(jitmodel))
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
