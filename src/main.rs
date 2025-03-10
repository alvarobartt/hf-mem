use anyhow::Context;

mod fetch;
use fetch::fetch_metadata;

mod token;
use token::get_token;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let url = "https://huggingface.co/gpt2/resolve/main/model.safetensors";
    let token = get_token().context("the hugging face auth token couldn't be retrieved")?;
    let metadata = fetch_metadata(url, &token)
        .await
        .context("failed to fetch metadata from the safetensors file from the hub")?;

    println!("metadata contains {metadata:?}");
    Ok(())
}
