extern crate tch;

use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use actix_web::middleware::{Compress, Logger};
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder, middleware, web, HttpRequest};
use actix_web::{web::{
    Data,
    Json,
}};
use actix_web::http::header;
use actix_web::http::header::ContentEncoding;
use actix_web::web::Query;
use serde::Serialize;
use serde::Deserialize;

#[derive(Debug, Deserialize, Serialize)]
pub struct V1RequestParams {
    item_ids: Vec<u64>,
    session_id: String,
}

#[post("/v1/recommend")]
async fn v1_recommend(
    app_data: web::Data<AppData>,
    query: Json<V1RequestParams>,
) -> impl Responder {
    let session_items: Vec<u64> = query.item_ids.clone();
    let device = app_data.device.as_ref();
    println!("device.is_cuda(): {}", device.is_cuda());
    let model = app_data.model.as_ref();
    let input = Tensor::rand(&[1, 1, 28, 28], (Kind::Float, *device));
    // Apply the model to the input tensor to perform inference
    let output = model.forward_ts(&[input]).unwrap();
    println!("{}", output);
    HttpResponse::Ok().json(vec![1,2,3,4,5,6,7])
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
    let path = std::env::args().nth(1).expect("no path to a model.pt given");
    let device = Device::cuda_if_available();

    // Load the model from the saved JIT script
    let model = Arc::new(tch::CModule::load_on_device(path, device).unwrap());
    let device = Arc::new(device);
    HttpServer::new(move || {
        let app_data = AppData {
            model: model.clone(),
            device: device.clone(),
        };
        App::new()
            .service(home)
            .service(ping)
            .service(v1_recommend)
            .app_data(Data::new(app_data))
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

//
// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//
//     let path = std::env::args().nth(1).expect("no path to a model.pt given");
//     let device = Device::cuda_if_available();
//     // Load the model from the saved JIT script
//     let model = Arc::new(tch::CModule::load_on_device(path, device).unwrap());
//
//     let bind_address = format!("{}:{}", "127.0.0.1", "7080");
//     println!("Done. start httpd at http://{}", &bind_address);
//     let num_workers = 2;
//     HttpServer::new(move || {
//         let handles_and_config = SharedHandlesAndConfig {
//             session_store: db.clone(),
//             vmis_index: vmis_index.clone(),
//             m_most_recent_sessions,
//             neighborhood_size_k,
//             num_items_to_recommend,
//             max_items_in_session,
//             qty_workers,
//             db_compaction_ttl_in_secs: session_ttl.as_secs() as usize,
//             enable_business_logic,
//             example_item_id,
//         };
//
//         App::new()
//             .wrap(middleware::Compress::new(ContentEncoding::Identity))
//             .wrap(prometheus.clone())
//             .wrap(
//                 middleware::DefaultHeaders::new()
//                     .header("Cache-Control", "no-cache, no-store, must-revalidate")
//                     .header("Pragma", "no-cache")
//                     .header("Expires", "0"),
//             )
//             // .data(handles_and_config)
//             .service(web::resource("/").route(web::get().to(|_req: HttpRequest| {
//                 HttpResponse::Found()
//                     .header(header::LOCATION, "/internal")
//                     .finish()
//             })))
//     })
//         .workers(num_workers)
//         .bind(&bind_address)
//         .unwrap_or_else(|_| panic!("Could not bind server to address {}", &bind_address))
//         .run()
//         .await
//
//     // // Create a random input tensor to demonstrate inference
//     // let input = Tensor::rand(&[1, 1, 28, 28], (Kind::Float, device));
//     //
//     // // Apply the model to the input tensor to perform inference
//     // let output = model.forward_ts(&[input]).unwrap();
//
//
// }