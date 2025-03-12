use anyhow::Context;
use clap::Parser;
use futures::future::join_all;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION},
    Client, StatusCode,
};
use serde::Deserialize;
use std::collections::HashMap;

mod fetch;
use fetch::{fetch, FileType};

mod token;
use token::get_token;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model_id: String,

    #[arg(short, long)]
    token: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ModelIndex {
    weight_map: std::collections::HashMap<String, String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let sharded_url = format!(
        "https://huggingface.co/{}/resolve/main/model.safetensors.index.json",
        args.model_id
    );
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

    let metadata = match client
        .get(sharded_url)
        .send()
        .await
        .context("fetching the sharded url failed")
    {
        Ok(response) => match response.status() {
            StatusCode::OK => {
                let model_index = serde_json::from_str::<ModelIndex>(
                    &response
                        .text()
                        .await
                        .context("failed to pull the text out of the response")?,
                )
                .context("failed deserializing the text into a json")?;

                let urls = model_index
                    .weight_map
                    .values()
                    .map(|v| {
                        format!(
                            "https://huggingface.co/{}/resolve/main/{}",
                            &args.model_id, v
                        )
                    })
                    .collect::<std::collections::HashSet<String>>();

                let mut tasks = Vec::new();
                for url in urls {
                    let ctoken = token.clone();
                    tasks.push(tokio::spawn(async move {
                        fetch(&url, &ctoken).await.context(
                            "failed to fetch metadata from the safetensors file from the hub",
                        )
                    }));
                }

                let mut metadata = HashMap::<String, FileType>::new();
                let futures = join_all(tasks).await;
                for future in futures {
                    match future {
                        Ok(Ok(result)) => {
                            metadata.extend(result);
                        }
                        Ok(Err(e)) => {
                            anyhow::bail!("failed to fetch: {:?}", e);
                        }
                        Err(e) => {
                            anyhow::bail!("failed to fetch: {:?}", e);
                        }
                    }
                }
                metadata
            }
            StatusCode::NOT_FOUND => {
                let url = format!(
                    "https://huggingface.co/{}/resolve/main/model.safetensors",
                    args.model_id
                );
                fetch(&url, &token)
                    .await
                    .context("failed to fetch metadata from the safetensors file from the hub")?
            }
            _ => anyhow::bail!("response {response:?}"),
        },
        Err(e) => anyhow::bail!("failed with error {e}"),
    };

    let mut parameters = std::collections::HashMap::<String, u64>::new();
    for value in metadata.values() {
        if value.shape.is_none() || value.dtype.is_none() {
            eprintln!("doesn't contain file shape {value:?}");
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
        let dtype = match k.split_once("_") {
            Some(ks) => ks.0,
            None => &k,
        };
        total_bytes += v * match dtype {
            "F32" => 4,
            "F16" | "BF16" => 2,
            "F8" => 1,
            _ => 0,
        };
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
