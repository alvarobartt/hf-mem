use anyhow::Context;
use byteorder::{ByteOrder, LittleEndian};
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, RANGE},
    Client, StatusCode,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let url = "https://huggingface.co/gpt2/resolve/main/model.safetensors";

    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str("Bearer hf_***")
            .context("parsing the authorization header with the hf token failed")?,
    );
    // As per https://github.com/huggingface/huggingface_hub/pull/1855#discussion_r1404286419, we
    // only fetch the first 100kb of the `model.safetensors` file, as empirically, 97% of
    // safetensors files have a metadata size < 100kb (over the top 1000 models on the Hub)
    headers.insert(
        RANGE,
        HeaderValue::from_str("bytes=0-100000")
            .context("parsing the range of bytes to fetch for the range header failed")?,
    );

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
        Ok(response) => {
            match response.status() {
                StatusCode::PARTIAL_CONTENT => {
                    let metadata = response
                        .bytes()
                        .await
                        .context("failed reading the bytes from the response")?;
                    let metadata_size = LittleEndian::read_u64(&metadata[..8]) as usize;
                    // TODO: if `metadata_size` over 100_000, then request the complete metadata
                    let metadata = serde_json::from_slice::<serde_json::Value>(
                        &metadata[8..metadata_size + 8],
                    )
                    .context("parsing the metadata bytes into a serde_json::Value failed")?;
                    println!("metadata contains {metadata:?}");
                    Ok(())
                }
                _ => anyhow::bail!("failed reading the file bytes"),
            }
        }
        Err(e) => anyhow::bail!("sending the get request to the provided url failed with {e}"),
    }
}
