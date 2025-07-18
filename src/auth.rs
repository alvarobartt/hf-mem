use std::path::PathBuf;

pub fn read_token_from_cache() -> anyhow::Result<String> {
    // If the HF_TOKEN hasn't been provided then try to read ~/.cache/huggingface/token, that
    // contains the authentication token set via the `huggingface-cli` or any other authentication
    // method
    if let Ok(home) = std::env::var("HOME") {
        let file = PathBuf::from(home).join(".cache/huggingface/token");
        if file.exists() {
            if let Ok(content) = std::fs::read_to_string(file) {
                if !content.is_empty() {
                    return Ok(content.trim().to_string());
                }
            }
        }
    };
    // If none of those return a result, then just fail, prompting the user to provide any of those
    anyhow::bail!(
        "Hugging Face authentication token not found. Please set the HF_TOKEN environment variable or run `huggingface-cli login` to authenticate."
    )
}
