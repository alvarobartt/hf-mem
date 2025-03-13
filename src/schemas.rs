use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug)]
pub struct FileType {
    pub dtype: Option<String>,
    pub shape: Option<Vec<u64>>,
    #[allow(unused)]
    data_offsets: Option<(u64, u64)>,
}

#[derive(Deserialize, Debug)]
pub struct ModelIndex {
    pub weight_map: HashMap<String, String>,
}
