use dirs::home_dir;

pub fn get_token() -> anyhow::Result<String> {
    // First, try to pull the authentication token for the Hugging Face Hub from the HF_TOKEN
    // environment variable (if available)
    if let Ok(token) = std::env::var("HF_TOKEN") {
        return Ok(token);
    };
    // Then, if the HF_TOKEN hasn't been provided then try to read ~/.cache/huggingface/token, that
    // contains the authentication token set via the `huggingface-cli` or any other authentication
    // method
    if let Some(home) = home_dir() {
        let file = home.join(".cache/huggingface/token");
        if file.exists() {
            if let Ok(content) = std::fs::read_to_string(file) {
                if !content.is_empty() {
                    return Ok(content);
                }
            }
        }
    };
    // If none of those return a result, then just fail, prompting the user to provide any of those
    anyhow::bail!(
        "Hugging Face authentication token not found. Please set the HF_TOKEN environment variable or run 'huggingface-cli login' to authenticate."
    )
}
