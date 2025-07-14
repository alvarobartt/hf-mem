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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "ID of the model on the Hugging Face Hub")]
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

    // TODO: move this somewhere else and improve the code a bit adding a bit more rationale to it,
    // and improving the efficiency of it
    let mut parameters = std::collections::HashMap::<String, u64>::new();
    for value in metadata.values() {
        if value.shape.is_none() || value.dtype.is_none() {
            continue;
        }
        let shape = value.shape.clone().unwrap();
        let layer_parameters = shape.iter().product::<u64>();
        let dtype = value.dtype.clone().unwrap();
        let parameter_count = parameters.entry(dtype).or_insert(0);
        *parameter_count += layer_parameters;
    }

    let mut total_bytes = 0;
    for (k, v) in parameters {
        // TODO: ideally to handle the dtype the user wants to serve the model in
        // note that for some models it might not work i.e. some models as
        // `deepseek-ai/DeepSeek-R1-0528` has multiple dtypes `F16, F8_E4M3, F32`
        let dtype_bytes = match k.as_ref() {
            "F64" | "I64" | "U64" => 8,
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "F8_E5M2" | "F8_E4M3" | "I8" | "U8" => 1,
            _ => 0,
        };
        if dtype_bytes != 0 {
            total_bytes += v * dtype_bytes;
        }
    }

    println!("Requirements to run inference with `{}`", args.model_id);
    println!(
        "  - Memory in MB: {:.2} MB",
        total_bytes as f64 / 1024_f64.powf(2_f64)
    );
    println!(
        "  - Memory in MB (+ 18% overhead): {:.2} MB",
        total_bytes as f64 / 1024_f64.powf(2_f64) * 1.18_f64
    );
    println!(
        "  - Memory in GiB: {:.2} GiB",
        total_bytes as f64 / 1024_f64.powf(3_f64)
    );
    println!(
        "  - Memory in GiB (+ 18% overhead): {:.2} GiB",
        total_bytes as f64 / 1024_f64.powf(3_f64) * 1.18_f64
    );

    Ok(())
}
