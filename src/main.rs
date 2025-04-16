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
    #[arg(short, long)]
    model_id: String,

    #[arg(short, long)]
    revision: Option<String>,

    #[arg(short, long)]
    token: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let token = if let Some(token) = args.token {
        token
    } else {
        get_token().context("the hugging face auth token couldn't be retrieved")?
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(format!("Bearer {}", token).as_str())
            .context("parsing the authorization header with the hf token failed")?,
    );

    let client = Client::builder()
        .default_headers(headers)
        .build()
        .context("couldn't build the reqwest client")?;

    let metadata = match fetch_sharded(&client, args.model_id.clone(), args.revision.clone()).await
    {
        Ok(metadata) => metadata,
        Err(e) => match e {
            RequestError::FileNotFound(..) => {
                match fetch(
                    &client,
                    args.model_id.clone(),
                    args.revision.clone(),
                    Some("model.safetensors".to_string()),
                )
                .await
                .context("also failed when fetching the consolidated safetensors file")
                {
                    Ok(metadata) => metadata,
                    Err(e) => anyhow::bail!(RequestError::Other(e)),
                }
            }
            _ => anyhow::bail!(RequestError::Other(e.into())),
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

    println!("Requirements to run inference with {}", args.model_id);
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
