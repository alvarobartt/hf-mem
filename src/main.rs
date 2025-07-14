use anyhow::Context;
use clap::Parser;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION},
    Client,
};

mod auth;
mod errors;
mod fetch;
mod schemas;

use auth::get_token;
use errors::RequestError;
use fetch::{fetch, fetch_sharded};

fn validate_model_id(model_id: &str) -> Result<String, String> {
    if model_id.is_empty() {
        return Err("Model ID cannot be empty".to_string());
    }
    if !model_id.contains('/') {
        return Err("Model ID must be in format 'username/model-name'".to_string());
    }
    let parts: Vec<&str> = model_id.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err("Model ID must be in format 'username/model-name'".to_string());
    }
    Ok(model_id.to_string())
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "ID of the model on the Hugging Face Hub", value_parser = validate_model_id)]
    model_id: String,

    #[arg(
        short,
        long,
        default_value = "main",
        help = "Revision of the model on the Hugging Face Hub"
    )]
    revision: Option<String>,

    #[arg(
        short,
        long,
        help = "Hugging Face Hub token with read access over the provided model ID, optional"
    )]
    token: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut headers = HeaderMap::new();

    let token = args.token.or_else(|| get_token().ok());
    if let Some(token) = &token {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(format!("Bearer {}", token).as_str())
                .context("Failed to parse authorization header with Hugging Face token")?,
        );
    };

    let client = Client::builder()
        .default_headers(headers)
        .build()
        .context("Failed to build HTTP client")?;

    let metadata = match fetch_sharded(&client, args.model_id.clone(), args.revision.clone()).await
    {
        Ok(metadata) => metadata,
        Err(e) => match e {
            RequestError::FileNotFound(..) => fetch(
                &client,
                args.model_id.clone(),
                args.revision.clone(),
                Some("model.safetensors".to_string()),
            )
            .await
            .context("Failed to fetch consolidated safetensors file")?,
            _ => return Err(e.into()),
        },
    };

    let mut total_bytes = 0u64;
    for value in metadata.values() {
        if let (Some(shape), Some(dtype)) = (&value.shape, &value.dtype) {
            let layer_parameters = shape.iter().product::<u64>();
            let dtype_bytes = match dtype.as_str() {
                "F64" | "I64" | "U64" => 8,
                "F32" | "I32" | "U32" => 4,
                "F16" | "BF16" | "I16" | "U16" => 2,
                "F8_E5M2" | "F8_E4M3" | "I8" | "U8" => 1,
                _ => continue,
            };
            total_bytes += layer_parameters * dtype_bytes;
        }
    }

    let mb = total_bytes as f64 / 1_048_576.0;
    let gib = total_bytes as f64 / 1_073_741_824.0;
    let overhead_factor = 1.18;

    println!("Requirements to run inference with `{}`", args.model_id);
    println!("  - Memory in MB: {:.2} MB", mb);
    println!(
        "  - Memory in MB (+ 18% overhead): {:.2} MB",
        mb * overhead_factor
    );
    println!("  - Memory in GiB: {:.2} GiB", gib);
    println!(
        "  - Memory in GiB (+ 18% overhead): {:.2} GiB",
        gib * overhead_factor
    );

    Ok(())
}
