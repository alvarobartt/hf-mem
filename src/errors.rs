use thiserror::Error;

#[derive(Error, Debug)]
pub enum RequestError {
    #[error("File `{0}` not found on the Hugging Face Hub.")]
    FileNotFound(String),

    #[error(
        "Authentication to the Hugging Face Hub failed, please make sure you're using a valid Hugging Face Hub Token, more information at https://huggingface.co/docs/hub/security-tokens."
    )]
    HubAuth,

    #[error(
        "The Hugging Face Hub seems to be down at the moment, make sure to check https://status.huggingface.co for updates."
    )]
    HubIsDown,

    #[error("Internal error, please make sure that the request is correct and/or retry later.")]
    Internal,

    #[error("Request failed due to unknown reasons with status code `{0}`.")]
    Unknown(reqwest::StatusCode),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
