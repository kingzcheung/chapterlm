use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub embedding_size: usize,
    pub num_hidden_layers: usize,
    pub num_hidden_groups: usize,
    pub num_attention_heads: usize,
    pub inner_group_num: usize,
    pub hidden_act: HiddenAct,
    pub intermediate_size: usize,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub classifier_dropout_prob: f64,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 21128,
            hidden_size: 312,
            embedding_size: 128,
            num_hidden_layers:4,
            num_hidden_groups: 1,
            num_attention_heads: 12,
            inner_group_num: 1,
            hidden_act: HiddenAct::Gelu,
            intermediate_size:1248,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob:  0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: 0.1,
            position_embedding_type: PositionEmbeddingType::Absolute,
        }
    }
}
