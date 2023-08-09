use std::net::Ipv4Addr;
use std::sync::Arc;
use std::time::Instant;
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
    pre_ms: f32,
    inf_ms: f32,
    post_ms: f32,
    mname: String,
    mthreads: i32,
    mdevice: String,

}

// https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#response_requirements
#[derive(Debug, Serialize)]
pub struct VertexResponse {
    pub items: Vec<Vec<i64>>,
    pub nf: NonFunctional,

}

#[post("/predictions/model/1.0/")]
async fn v1_recommend(
    models: Data<Models>,
    query: Json<VertexRequest>,
) -> impl Responder {
    let preprocess_start_time = Instant::now();
    let session_items: Vec<i64> = query.instances.get(0).unwrap().context.clone();
    let preprocess_ms = preprocess_start_time.elapsed().as_millis();

    let inference_start_time = Instant::now();
    let (result_item_ids,  model_filename, model_qty_threads, model_device) : (Vec<i64>, String, i32, String) = match (&*models.jitopt_model, &*models.onnx_model) {
        (Some(ref model), None) => {
            println!("JITModelRuntime");
            (model.recommend(&session_items), model.get_model_filename(), model.get_model_qty_threads(), model.get_model_device_name())
        }
        (None, Some(ref model)) => {
            println!("OnnxModelRuntime");
            (model.recommend(&session_items), model.get_model_filename(), model.get_model_qty_threads(), model.get_model_device_name())
        }
        _ => {
            println!("DummyModelRuntime");
            let model = models.dummy_model.as_ref();
            (model.recommend(&session_items), model.get_model_filename(), model.get_model_qty_threads(), model.get_model_device_name())
        },
    };
    let inference_ms = inference_start_time.elapsed().as_millis();

    let response = &VertexResponse {
        items: vec![result_item_ids],
        nf: NonFunctional { pre_ms: preprocess_ms as f32, inf_ms: inference_ms as f32, post_ms: 0.0, mname: model_filename, mthreads: model_qty_threads, mdevice: model_device },
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
    pub jitopt_model: Arc<Option<JITModelRuntime>>,
    pub onnx_model: Arc<Option<OnnxModelRuntime>>,
    pub dummy_model: Arc<DummyModelRuntime>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    host: Ipv4Addr,
    port: u16,
    qty_actix_workers: usize,
    qty_model_threads: usize,
    model_path: String,
    payload_path: String,
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

    let qty_actix_threads = config.qty_actix_workers;
    let qty_model_threads = config.qty_model_threads;
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

    let jitmodelruntime: Arc<Option<JITModelRuntime>> = if config.model_path.ends_with("_jitopt.pth") {
        println!("loading JIT model from: {0}", config.model_path);
        Arc::new(Some(JITModelRuntime::new(&config.model_path, &config.payload_path, &qty_model_threads)))
    } else{
        Arc::new(None)
    };
    let onnxruntime: Arc<Option<OnnxModelRuntime>> = if config.model_path.ends_with("_onnx.pth") {
        println!("loading ONNX model from: {0}", config.model_path);
        Arc::new(Some(OnnxModelRuntime::new(&config.model_path, &config.payload_path, &qty_model_threads)))
    } else{
        Arc::new(None)
    };
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
