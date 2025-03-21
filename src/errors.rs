use thiserror::Error;

#[derive(Error, Debug)]
pub enum HubError {
    #[error("file `{0}` not found on the hub")] // 404
    FileNotFound(String),
    #[error("hub seems to be unavailable at the moment, retry later")] // 503
    HubIsDown,
    #[error("hub authentication failed, make sure that the hf token is valid")] // 403
    HubAuth,
    #[error("internal error, make sure that the request is correct and retry later")] // 500
    Internal,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
