use anyhow::Context;
use clap::Parser;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION},
    Client,
};

mod errors;
mod fetch;
mod schemas;
mod sharded;
mod token;

use fetch::fetch;
use sharded::fetch_sharded;
use token::get_token;

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

    let metadata = match fetch_sharded(&client, args.model_id.clone(), args.revision.clone()).await.context("failed while fetching the safetensors sharded, rolling back to consolidated safetensors (if available)") {
        Ok(metadata) => metadata,
        // TODO: the error should indeed contain the status code so as to handle it properly or
        // just link thiserror with the anyhow bail?
        Err(..) => {
            // TODO: find a mock proposal below:
            // match e {
            //      HubError::FileNotFound(..) => match fetch { ... },
            //      HubError::HubIsDown | HubError::HubUnavailable | HubError::HubAuthFailed => bail!
            // }
            match fetch(&client, args.model_id.clone(), args.revision.clone(), Some("model.safetensors".to_string())).await.context("also failed when fetching the consolidated safetensors file") {
                Ok(metadata) => metadata,
                Err(e) => anyhow::bail!(e),
            }
        }
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
        // TODO: create some custom enums for the dtypes rather than having a match here
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
