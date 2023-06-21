use std::net::Ipv4Addr;
use std::sync::Arc;
use actix_web::middleware::{Logger};
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder, middleware, http::header};
use actix_web::{web::{
    Data,
    Json,
}};
use serde::{Deserialize, Serialize};
use serving::modelruntime::dummyruntime::DummyRuntime;
use serving::modelruntime::ModelEngine;

// https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#prediction
#[derive(Debug, Deserialize, Serialize)]
pub struct VertexRequest {
    instances: Vec<VertexRequestContext>,
    parameters: Vec<VertexRequestParameter>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VertexRequestContext {
    context: Vec<i64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VertexRequestParameter {
    runtime: String,
}

#[derive(Debug, Serialize)]
pub struct NonFunctional {
    preprocess_ms: f32,
    inference_ms: f32,
    postprocess_ms: f32,
    model: String,
    device: String,
}

// https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#response_requirements
#[derive(Debug, Serialize)]
pub struct VertexResponse {
    pub items: Vec<Vec<i64>>,
    pub nf: NonFunctional,

}

#[post("/predictions/model")]
async fn v1_recommend(
    models: Data<Models>,
    query: Json<VertexRequest>,
) -> impl Responder {
    let session_items: Vec<i64> = query.instances.get(0).unwrap().context.clone();

    let result_item_ids = models.model.recommend(&session_items);
    let response = &VertexResponse {
        items: vec![result_item_ids],
        nf: NonFunctional {
            preprocess_ms: 0.0,
            inference_ms: 0.0,
            postprocess_ms: 0.0,
            model: "".to_string(),
            device: "".to_string(),
        },
    };
    HttpResponse::Ok().json(response)
}


#[derive(Serialize)]
pub struct GenericResponse {
    pub status: String,
    pub message: String,
}

#[get("/ping")]
async fn ping() -> impl Responder {
    const MESSAGE: &str = "EtudeServing is running";

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
    pub model: Arc<DummyRuntime>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    host: Ipv4Addr,
    port: u16,
    model_store_path: String,
    model_filename: String,
    payload_filename: String,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let qty_logical_cores = num_cpus::get();
    let qty_physical_cores = num_cpus::get_physical();
    println!("Number of logical cores: {qty_logical_cores}");
    println!("Number of physical cores: {qty_physical_cores}");
    let qty_actix_threads = qty_physical_cores;
    // let (qty_actix_threads, qty_model_threads) = if qty_logical_cores == 1 {
    //     (1, 1)
    // } else {
    //     let division = qty_logical_cores / 4;
    //     let remainder = qty_logical_cores - division;
    //     if remainder == 0 {
    //         (1, 1)
    //     } else {
    //         (3, 3)
    //     }
    // };
    println!("Number of actix threads: {qty_actix_threads}");

    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "error");
    }
    env_logger::init();
    println!("EtudeServing started successfully");

    let model = Arc::new(DummyRuntime::new());
    HttpServer::new(move || {
        let models = Models {
            model: model.clone(),
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
        .bind(("0.0.0.0", 8080)).unwrap_or_else(|_| panic!("Could not bind server to address"))
        .workers(qty_actix_threads)
        .run()
        .await
}
