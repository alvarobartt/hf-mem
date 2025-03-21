// TODO: add a progress bar and limit the number of concurrent requests to a reasonable default,
// rather than the current which is not limited and may cause bottlenecks
use anyhow::Context;
use byteorder::{ByteOrder, LittleEndian};
use reqwest::{
    header::{HeaderValue, RANGE},
    Client, StatusCode,
};
use std::collections::HashMap;

use crate::errors::HubError;
use crate::schemas::FileType;

// As per https://github.com/huggingface/huggingface_hub/pull/1855#discussion_r1404286419, we
// only fetch the first 100kb of the `model.safetensors` file, as empirically, 97% of
// safetensors files have a metadata size < 100kb (over the top 1000 models on the Hub)
static MAX_METADATA_SIZE: usize = 100_000;

async fn fetch_metadata(
    client: &Client,
    url: &str,
    range_lower: Option<usize>,
    range_upper: Option<usize>,
) -> anyhow::Result<(bytes::Bytes, usize), HubError> {
    let range_lower = range_lower.unwrap_or(0);
    let range_upper = range_upper.unwrap_or(MAX_METADATA_SIZE);

    match client
        .get(url)
        .header(
            RANGE,
            HeaderValue::from_str(format!("bytes={}-{}", range_lower, range_upper).as_str())
                .context("parsing the range of bytes to fetch for the range header failed")?,
        )
        .send()
        .await
        .context("fetching the provided url failed")
    {
        Ok(response) => match response.status() {
            StatusCode::PARTIAL_CONTENT => {
                let metadata = response
                    .bytes()
                    .await
                    .context("failed reading the bytes from the response")?;
                let metadata_size = LittleEndian::read_u64(&metadata[..8]) as usize;
                Ok((metadata, metadata_size))
            }
            StatusCode::NOT_FOUND => Err(HubError::FileNotFound(url.to_string())),
            _ => Err(HubError::Internal),
        },
        Err(e) => Err(HubError::Other(e)),
    }
}

pub async fn fetch(
    client: &Client,
    model_id: String,
    revision: Option<String>,
    filename: Option<String>,
) -> anyhow::Result<HashMap<String, FileType>, HubError> {
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        model_id,
        revision.unwrap_or("main".to_string()),
        filename.unwrap_or("model.safetensors".to_string()),
    );

    let (metadata, metadata_size) = fetch_metadata(&client, &url, None, None)
        .await
        .context("failed fetching the metadata from the provided url")?;

    let metadata = if metadata_size > MAX_METADATA_SIZE {
        fetch_metadata(&client, &url, Some(8), Some(metadata_size))
            .await
            .context("failed fetching the complete metadata from the provided url")?
            .0
    } else {
        metadata
    };

    Ok(
        serde_json::from_slice::<HashMap<String, FileType>>(&metadata[8..metadata_size + 8])
            .context("parsing the metadata bytes into a serde_json::Value failed")?,
    )
}
