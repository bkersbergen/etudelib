use std::sync::Arc;
use actix_web::middleware::{Logger};
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder, middleware, http::header};
use actix_web::{web::{
    Data,
    Json,
}};
use serde::{Deserialize, Serialize};
use serving::modelruntime::jitmodelruntime::JITModelRuntime;
use serving::modelruntime::ModelEngine;
use serving::modelruntime::onnxmodelruntime::OnnxModelRuntime;


#[derive(Debug, Deserialize, Serialize)]
pub struct V1RequestParams {
    item_ids: Vec<i64>,
    session_id: String,
}


#[post("/v1/recommend")]
async fn v1_recommend(
    models: Data<Models>,
    query: Json<V1RequestParams>,
) -> impl Responder {
    let session_items: Vec<i64> = query.item_ids.clone();

    let response = match (&*models.jitopt_model, &*models.onnx_model) {
        (Some(ref model), None) => {
            let result_item_ids = model.recommend(&session_items);
            HttpResponse::Ok().json(result_item_ids)
        }
        (None, Some(ref model)) => {
            let result_item_ids = model.recommend(&session_items);
            HttpResponse::Ok().json(result_item_ids)
        }
        _ => {
            HttpResponse::Ok().json(vec![-1])
        },
    };
    return response;
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

pub struct Models {
    pub jitopt_model: Arc<Option<JITModelRuntime>>,
    pub onnx_model: Arc<Option<OnnxModelRuntime>>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "error");
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
    let onnxruntime: Arc<Option<OnnxModelRuntime>> = if model_path.ends_with("_onnx.pth") {
        Arc::new(Some(OnnxModelRuntime::new(&model_path, &payload_path)))
    } else{
        Arc::new(None)
    };
    HttpServer::new(move || {
        let models = Models {
            jitopt_model: jitmodelruntime.clone(),
            onnx_model: onnxruntime.clone(),
        };
        App::new()
            .service(home)
            .service(ping)
            .service(v1_recommend)
            .app_data(Data::new(models))
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