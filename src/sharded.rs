use anyhow::Context;
use futures::future::join_all;
use reqwest::{Client, StatusCode};
use std::collections::{HashMap, HashSet};

use crate::errors::HubError;
use crate::fetch::fetch;
use crate::schemas::{FileType, ModelIndex};

pub async fn fetch_sharded(
    client: &Client,
    model_id: String,
    revision: Option<String>,
) -> anyhow::Result<HashMap<String, FileType>, HubError> {
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/model.safetensors.index.json",
        model_id,
        revision.clone().unwrap_or("main".to_string()),
    );

    match client
        .get(&url)
        .send()
        .await
        .context("fetching the sharded url failed")
    {
        Ok(response) => match response.status() {
            StatusCode::OK => {
                let model_index = serde_json::from_str::<ModelIndex>(
                    &response
                        .text()
                        .await
                        .context("failed to pull the text out of the response")?,
                )
                .context("failed deserializing the text into a json")?;

                let filenames = model_index
                    .weight_map
                    .values()
                    .cloned()
                    .collect::<HashSet<String>>();

                let mut tasks = Vec::new();
                for filename in filenames {
                    let client_clone = client.clone();
                    let model_id_clone = model_id.clone();
                    let revision_clone = revision.clone();
                    tasks.push(tokio::spawn(async move {
                        fetch(
                            &client_clone,
                            model_id_clone,
                            revision_clone,
                            Some(filename),
                        )
                        .await
                        .context("failed to fetch metadata from the safetensors file from the hub")
                    }));
                }

                let mut metadata = HashMap::<String, FileType>::new();
                let futures = join_all(tasks).await;
                for future in futures {
                    match future {
                        Ok(Ok(result)) => {
                            metadata.extend(result);
                        }
                        Ok(Err(..)) => panic!("failed capturing future"),
                        Err(..) => panic!("failed capturing future"),
                    }
                }
                Ok(metadata)
            }
            StatusCode::NOT_FOUND => Err(HubError::FileNotFound(
                "`model.safetensors.index.json`".to_string(),
            )),
            StatusCode::FORBIDDEN => Err(HubError::HubAuth),
            _ => Err(HubError::Internal),
        },
        Err(err) => Err(HubError::Other(err)),
    }
}
