extern crate tch;

use std::sync::Arc;
use tch::{Device, Tensor};
use actix_web::middleware::{Logger};
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder, middleware, http::header};
use actix_web::{web::{
    Data,
    Json,
}};
use serde::{Deserialize, Serialize};
use serde_yaml::{self};
use serving::modelruntime::jitmodelruntime::JITModelRuntime;
use serving::modelruntime::ModelEngine;


#[derive(Debug, Deserialize, Serialize)]
pub struct V1RequestParams {
    item_ids: Vec<i64>,
    session_id: String,
}


#[post("/v1/recommend")]
async fn v1_recommend(
    jitmodel: Data<JITModel>,
    query: Json<V1RequestParams>,
) -> impl Responder {
    let session_items: Vec<i64> = query.item_ids.clone();

    match &*jitmodel.modelruntime {
        Some(ref model) => {
            let result_item_ids = model.recommend(&session_items);
            HttpResponse::Ok().json(result_item_ids)
        }
        None => {
            HttpResponse::Ok().json(vec![0])
        }
    }
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
        .append_header((header::LOCATION, "/ping"))
        .finish()
}

pub struct JITModel {
    pub modelruntime: Arc<Option<JITModelRuntime>>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "debug");
    }
    env_logger::init();
    println!("Actix Server started successfully");
    let model_path = std::env::args().nth(1).expect("no path to a model.pt given");
    let payload_path = std::env::args().nth(2).expect("no path to payload.yaml given");

    let jitmodelruntime: Arc<Option<JITModelRuntime>> = if model_path.ends_with("_jitopt.pth") {
        Arc::new(Some(JITModelRuntime::new(&model_path, &payload_path)))
    } else{
        Arc::new(None)
    };
    if model_path.ends_with("_onnx.pth") {
        println!("Loading onnx model");
    }
    HttpServer::new(move || {
        let jitmodel = JITModel {
            modelruntime: jitmodelruntime.clone(),
        };
        App::new()
            .service(home)
            .service(ping)
            .service(v1_recommend)
            .app_data(Data::new(jitmodel))
            .wrap(Logger::default())
            .wrap(
                middleware::DefaultHeaders::new()
                    .add(("Cache-Control", "no-cache, no-store, must-revalidate"))
                    .add(("Pragma", "no-cache"))
                    .add(("Expires", "0")),
            )
    })
        .bind(("127.0.0.1", 7080)).unwrap_or_else(|_| panic!("Could not bind server to address"))
        .run()
        .await
}
