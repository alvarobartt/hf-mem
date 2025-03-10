use anyhow::Context;
use clap::Parser;
use serde::Deserialize;

mod fetch;
use fetch::fetch;

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

// TODO: add an enum for the dtypes
// Python reference is https://github.com/huggingface/huggingface_hub/blob/c51782a913d128fdab12f065e7a2de20fac3b7d3/src/huggingface_hub/utils/_safetensors.py
#[derive(Deserialize, Debug)]
struct FileType {
    dtype: String,
    shape: Vec<u64>,
    // data_offsets: (u64, u64),
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let url = format!(
        "https://huggingface.co/{}/resolve/main/model.safetensors",
        args.model_id
    );

    let token = if let Some(token) = args.token {
        token
    } else {
        get_token().context("the hugging face auth token couldn't be retrieved")?
    };

    let metadata = fetch(&url, &token)
        .await
        .context("failed to fetch metadata from the safetensors file from the hub")?;

    // TODO: this here is too slow, probably due to the `serde_json` conversions etc.
    let mut parameters = std::collections::HashMap::<String, u64>::new();
    for value in metadata
        .as_object()
        .context("failed to parse metadata json as an object")?
        .values()
    {
        // TODO: probably we can just avoid the clone here
        let file_type: FileType = serde_json::from_value(value.clone())
            .context("failed to deserialize json value into file type")?;
        let layer_parameters = file_type.shape.iter().product::<u64>();
        let parameter_count = parameters.entry(file_type.dtype).or_insert(0);
        *parameter_count += layer_parameters;
    }
    let mut total_bytes = 0;
    for (k, v) in parameters {
        // TODO: ideally use a match to handle all the dtypes
        total_bytes += if k == "F32" { v * 4 } else { 0 };
    }
    println!("total required {total_bytes} B");
    let total_mb = total_bytes as f64 / 1024_f64.powf(2_f64);
    println!("total required {total_mb:.2} MB");
    let total_gb = total_bytes as f64 / 1024_f64.powf(3_f64);
    println!("total required {total_gb:.5} GB");
    println!(
        "total required giga bytes including the extra memory estimation of an extra {} % (for CUDA Graphs, KV-cache, etc.) is {:.5} GB",
        18_usize,
        total_gb * 1.18_f64,
    );

    Ok(())
}
