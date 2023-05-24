use goose::prelude::*;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use serde_json::json;
use rand::distributions::Distribution;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;


#[tokio::main]
async fn main() -> Result<(), GooseError> {

    let endpoint_url_arg = env::var("VERTEX_ENDPOINT_URL").unwrap_or_else(|_| {
        eprintln!("Error: The 'VERTEX_ENDPOINT_URL' environment variable is not defined.");
        std::process::exit(1);
    });
    println!("ENV_VAR[VERTEX_ENDPOINT_URL] = {endpoint_url_arg}");
    let catalog_size_arg: i32= env::var("CATALOG_SIZE").unwrap_or_else(|_| {
        eprintln!("Error: The 'CATALOG_SIZE' environment variable is not defined.");
        std::process::exit(1);
    }).parse().unwrap();
    println!("ENV_VAR[CATALOG_SIZE] = {catalog_size_arg}");
    let report_location_arg = env::var("REPORT_LOCATION").unwrap_or_else(|_| {
        eprintln!("Error: The 'REPORT_LOCATION' environment variable is not defined.");
        std::process::exit(1);
    });
    println!("ENV_VAR[REPORT_LOCATION] = {report_location_arg}");

    let dummy_variable: bool = true;

    let closure: TransactionFunction = Arc::new(move |user| {
        Box::pin(async move {
            // Call the recommend function with the custom variable.
            recommend(user, dummy_variable).await?;
            Ok(())
        })
    });
    let transaction = Transaction::new(closure);
    GooseAttack::initialize()?
        .register_scenario(
            scenario!("PyTorch")
                .register_transaction(transaction)
        )
        .execute()
        .await?;

    Ok(())
}

async fn recommend(user: &mut GooseUser, _dummy_variable: bool) -> TransactionResult {
    let mut rng = StdRng::from_entropy();
    let session_length: i32 = rng.gen_range(1..15);
    let item_ids = (0..session_length).map(|_| rng.gen_range(1..1000)).collect::<Vec<i64>>();
    let payload = json!({"instances": [{"context": item_ids}],"parameters": [{"runtime":  ""}]});

    let request_builder = user.get_request_builder(&GooseMethod::Post, "/v1/recommend")?
        // Configure the request to timeout if it takes longer than xxx milliseconds.
        .header("Content-Type", "application/json")
        .body(payload.to_string())
        .timeout(Duration::from_millis(500));

    // Manually build a GooseRequest.
    let goose_request = GooseRequest::builder()
        // Manually add our custom RequestBuilder object.
        .set_request_builder(request_builder)
        // Turn the GooseRequestBuilder object into a GooseRequest.
        .build();

    let response = user.request(goose_request).await?;
    // let recos = response.response.unwrap().json::<Vec<i64>>().await.unwrap();
    // println!("{:?}", recos);
    Ok(())
}
