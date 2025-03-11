use anyhow::Context;
use byteorder::{ByteOrder, LittleEndian};
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, RANGE},
    Client, StatusCode,
};

// As per https://github.com/huggingface/huggingface_hub/pull/1855#discussion_r1404286419, we
// only fetch the first 100kb of the `model.safetensors` file, as empirically, 97% of
// safetensors files have a metadata size < 100kb (over the top 1000 models on the Hub)
static MAX_METADATA_SIZE: usize = 100_000;

// TODO: we need to first check if the files are available in the Hugging Face cache, if so, just read those directly; if not, then fetch those from the Hub (probably after the first release)
// TODO: in some cases there will be a `model.safetensors.index.json` instead of a single
// `model.safetensors` file, and the different sharded files need to be read
async fn fetch_metadata(
    url: &str,
    token: &str,
    range_lower: Option<usize>,
    range_upper: Option<usize>,
) -> anyhow::Result<(bytes::Bytes, usize)> {
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(format!("Bearer {}", token).as_str())
            .context("parsing the authorization header with the hf token failed")?,
    );
    let range_lower = range_lower.unwrap_or(0);
    let range_upper = range_upper.unwrap_or(MAX_METADATA_SIZE);
    headers.insert(
        RANGE,
        HeaderValue::from_str(format!("bytes={}-{}", range_lower, range_upper).as_str())
            .context("parsing the range of bytes to fetch for the range header failed")?,
    );

    // TODO: do we really need a client?
    let client = Client::builder()
        .default_headers(headers)
        .build()
        .context("couldn't build the reqwest client")?;

    match client
        .get(url)
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
            _ => anyhow::bail!("failed reading the file bytes"),
        },
        Err(e) => anyhow::bail!("sending the get request to the provided url failed with {e}"),
    }
}

pub async fn fetch(url: &str, token: &str) -> anyhow::Result<serde_json::Value> {
    let (metadata, metadata_size) = fetch_metadata(url, token, None, None)
        .await
        .context("failed fetching the metadata from the provided url")?;

    let metadata = if metadata_size > MAX_METADATA_SIZE {
        fetch_metadata(url, token, Some(8), Some(metadata_size))
            .await
            .context("failed fetching the complete metadata from the provided url")?
            .0
    } else {
        metadata
    };

    serde_json::from_slice::<serde_json::Value>(&metadata[8..metadata_size + 8])
        .context("parsing the metadata bytes into a serde_json::Value failed")
}
