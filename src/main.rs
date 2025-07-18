use anyhow::Context;
use clap::Parser;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION},
    Client,
};
use std::collections::HashSet;

mod auth;
mod errors;
mod fetch;
mod schemas;

use auth::read_token_from_cache;
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

fn validate_dtype(dtype: &str) -> Result<String, String> {
    match dtype {
        "float32" | "float16" | "bfloat16" | "float8" | "float4" => Ok(dtype.to_string()),
        _ => Err(
            "Invalid dtype. Must be one of: float32, float16, bfloat16, float8, float4".to_string(),
        ),
    }
}

fn get_dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "float32" => 4,
        "float16" | "bfloat16" => 2,
        "float8" => 1,
        "float4" => 1, // 0.5 bytes per parameter, but handled separately
        _ => 4,
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, env, help = "ID of the model on the Hugging Face Hub", value_parser = validate_model_id)]
    model_id: String,

    #[arg(
        short,
        long,
        env,
        default_value = "main",
        help = "Revision of the model on the Hugging Face Hub"
    )]
    revision: Option<String>,

    #[arg(
        short,
        long,
        env = "HT_TOKEN",
        help = "Hugging Face Hub token with read access over the provided model ID, optional"
    )]
    token: Option<String>,

    #[arg(
        short,
        long,
        env,
        help = "Target dtype for conversion (float32, float16, bfloat16, float8, float4)",
        value_parser = validate_dtype
    )]
    dtype: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut headers = HeaderMap::new();

    let token = args.token.or_else(|| read_token_from_cache().ok());
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

    let mut dtype_counts = std::collections::HashMap::new();
    let mut detected_dtypes = HashSet::new();
    let mut total_bytes = 0u64;

    for value in metadata.values() {
        if let (Some(shape), Some(dtype)) = (&value.shape, &value.dtype) {
            let layer_parameters = shape.iter().product::<u64>();
            detected_dtypes.insert(dtype.clone());
            *dtype_counts.entry(dtype.clone()).or_insert(0u64) += layer_parameters;

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

    let (converted_bytes, conversion_applied) = if let Some(target_dtype) = &args.dtype {
        if detected_dtypes.len() == 1 {
            let total_params: u64 = dtype_counts.values().sum();
            let target_bytes = if target_dtype == "float4" {
                // float4 uses 0.5 bytes per parameter
                (total_params + 1) / 2 // Round up for odd number of parameters
            } else {
                total_params * get_dtype_bytes(target_dtype)
            };
            (target_bytes, true)
        } else {
            println!(
                "Warning: Model contains multiple dtypes: {:?}",
                detected_dtypes
            );
            println!("Dtype conversion not applied for multi-dtype models.");
            println!(
                "Multi-dtype models typically use different precisions for different layer types:"
            );
            println!("- Attention layers often use higher precision (F16/BF16)");
            println!("- Feed-forward layers may use lower precision (F8/F4)");
            println!("- Embeddings usually maintain higher precision for quality");
            println!(
                "Converting all layers to the same dtype may impact model quality significantly."
            );
            (total_bytes, false)
        }
    } else {
        (total_bytes, false)
    };

    let display_bytes = if conversion_applied {
        converted_bytes
    } else {
        total_bytes
    };

    let mb = display_bytes as f64 / 1_048_576.0;
    let gib = display_bytes as f64 / 1_073_741_824.0;
    let overhead_factor = 1.18;

    let model_desc = if let Some(target_dtype) = &args.dtype {
        if conversion_applied {
            format!("{} (converted to {})", args.model_id, target_dtype)
        } else {
            format!("{} (original dtypes)", args.model_id)
        }
    } else {
        args.model_id.clone()
    };

    println!("Requirements to run inference with `{}`", model_desc);
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
