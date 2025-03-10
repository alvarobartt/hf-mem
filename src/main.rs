use anyhow::Context;
use clap::Parser;

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
    println!("metadata contains {metadata:?}");

    Ok(())
}
