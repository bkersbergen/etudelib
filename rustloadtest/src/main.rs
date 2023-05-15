use goose::prelude::*;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), GooseError> {

    // Check if the "sync" command line argument is provided
    let args: Vec<String> = env::args().collect();
    // let synchronous_mode = args.len() > 1 && args[1] == "sync";
    let synchronous_mode: bool = true;
    println!("synchronous_mode: {synchronous_mode}");

    let closure: TransactionFunction = Arc::new(move |user| {
        Box::pin(async move {
            // Call the recommend function with the custom variable.
            recommend(user, synchronous_mode).await?;
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

async fn recommend(user: &mut GooseUser, synchronous_mode: bool) -> TransactionResult {
    let payload = json!({
        "item_ids": [1, 5, 3, 1, 7, 1],
        "session_id": "abcdefg"
    });

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

    if synchronous_mode {
        let response = user.request(goose_request).await?;
        // let recos = response.response.unwrap().json::<Vec<i64>>().await.unwrap();
        // println!("{:?}", recos);
    } else {
        let future = user.request(goose_request);
    };
    Ok(())
}
