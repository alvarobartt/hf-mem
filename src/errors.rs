use thiserror::Error;

#[derive(Error, Debug)]
pub enum RequestError {
    #[error("File '{0}' not found on Hugging Face Hub")]
    FileNotFound(String),

    #[error(
        "Authentication failed: Invalid or missing Hugging Face token. Please ensure you have a valid token from https://huggingface.co/settings/tokens"
    )]
    HubAuth,

    #[error(
        "Hugging Face Hub is currently unavailable. Please check https://status.huggingface.co for service status"
    )]
    HubIsDown,

    #[error("Internal server error occurred. Please verify your request and try again")]
    Internal,

    #[error("Request failed with status code {0}")]
    Unknown(reqwest::StatusCode),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("JSON parsing error: {0}")]
    JsonParse(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
