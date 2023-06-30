use std::net::Ipv4Addr;
use std::sync::Arc;
use actix_web::middleware::{Logger};
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder, middleware, http::header};
use actix_web::{web::{
    Data,
    Json,
}};
use serde::{Deserialize, Serialize};
use serving::modelruntime::ModelEngine;
use serving::modelruntime::dummymodelruntime::DummyModelRuntime;
use serving::modelruntime::jitmodelruntime::JITModelRuntime;
use serving::modelruntime::onnxmodelruntime::OnnxModelRuntime;


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

#[post("/predictions/model/1.0")]
async fn v1_recommend(
    models: Data<Models>,
    query: Json<VertexRequest>,
) -> impl Responder {
    let session_items: Vec<i64> = query.instances.get(0).unwrap().context.clone();

    let result_item_ids :Vec<i64> = match (&*models.jitopt_model, &*models.onnx_model) {
        (Some(ref model), None) => {
            model.recommend(&session_items)
        }
        (None, Some(ref model)) => {
            model.recommend(&session_items)
        }
        _ => {
            let model = models.dummy_model.as_ref();
            model.recommend(&session_items)
        },
    };
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
    return response;
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
    pub jitopt_model: Arc<Option<JITModelRuntime>>,
    pub onnx_model: Arc<Option<OnnxModelRuntime>>,
    pub dummy_model: Arc<DummyModelRuntime>,
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
    let arg_configfile = std::env::args().nth(1).expect("no path to a server configfile.yaml given");
    let f = std::fs::File::open(arg_configfile).expect("Could not open configfile.");
    let config: Config = serde_yaml::from_reader(f).expect("Could not read values.");
    println!("{:?}", config);

    let qty_logical_cores = num_cpus::get();
    let qty_physical_cores = num_cpus::get_physical();
    println!("Number of logical cores: {qty_logical_cores}");
    println!("Number of physical cores: {qty_physical_cores}");
    let qty_actix_threads = qty_physical_cores;
    let qty_model_threads = 1;
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
    println!("Number of model threads: {qty_model_threads}");

    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "error");
    }
    env_logger::init();
    println!("EtudeServing started successfully");

    let jitmodelruntime: Arc<Option<JITModelRuntime>> = if config.model_filename.ends_with("_jitopt.pth") {
        let model_path = format!("{}{}{}", &config.model_store_path, std::path::MAIN_SEPARATOR, &config.model_filename);
        let payload_path = format!("{}{}{}", &config.model_store_path, std::path::MAIN_SEPARATOR, &config.payload_filename);
        Arc::new(Some(JITModelRuntime::new(&model_path, &payload_path, &qty_model_threads)))
    } else{
        Arc::new(None)
    };
    let onnxruntime: Arc<Option<OnnxModelRuntime>> = if config.model_filename.ends_with("_onnx.pth") {
        let model_path = format!("{}{}{}", &config.model_store_path, std::path::MAIN_SEPARATOR, &config.model_filename);
        let payload_path = format!("{}{}{}", &config.model_store_path, std::path::MAIN_SEPARATOR, &config.payload_filename);
        Arc::new(Some(OnnxModelRuntime::new(&model_path, &payload_path, &qty_model_threads)))
    } else{
        Arc::new(None)
    };
    assert!(jitmodelruntime.is_some() || onnxruntime.is_some(), "Both JITModelRuntime and OnnxModelRuntime are None.");
    let dummyruntime: Arc<DummyModelRuntime> = Arc::new(DummyModelRuntime::new());

    HttpServer::new(move || {
        let models = Models {
            jitopt_model: jitmodelruntime.clone(),
            onnx_model: onnxruntime.clone(),
            dummy_model: dummyruntime.clone(),
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
        .bind((config.host, config.port)).unwrap_or_else(|_| panic!("Could not bind server to address"))
        .workers(qty_actix_threads)
        .run()
        .await
}
