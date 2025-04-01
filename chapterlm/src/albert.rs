use candle_core::{D, DType, IndexOp, Result, Tensor};
use candle_nn::{
    Activation, Dropout, Embedding, LayerNorm, Linear, Module, VarBuilder, activation, embedding,
    layer_norm, linear,
};

use crate::{
    config::{Config, HiddenAct, PositionEmbeddingType},
    util::{_prepare_4d_attention_mask_for_sdpa, prune_linear_layer, scaled_dot_product_attention},
};

#[derive(Debug, Clone)]
pub struct AlbertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    position_embedding_type: PositionEmbeddingType,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl AlbertEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.embedding_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.embedding_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.embedding_size,
            vb.pp("token_type_embeddings"),
        )?;

        let layer_norm = layer_norm(
            config.embedding_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            position_embedding_type: config.position_embedding_type,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        //embeddings = inputs_embeds + token_type_embeddings
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;

            let position_embeddings = position_embeddings.forward(&position_ids)?;
            embeddings = embeddings.broadcast_add(&position_embeddings)?
        }

        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings, false)?;

        Ok(embeddings)
    }
}

#[derive(Debug, Clone)]
struct AlbertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    hidden_size: usize,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,

    attention_dropout: Dropout,
    output_dropout: Dropout,
    dense: Linear,
    layer_norm: LayerNorm,

    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl AlbertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;

        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            query,
            key,
            value,
            hidden_size,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            attention_dropout: Dropout::new(config.attention_probs_dropout_prob),
            output_dropout: Dropout::new(config.hidden_dropout_prob),
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    // 剪枝
    fn prune(&self, _heads: &[usize]) -> Result<()> {
        unimplemented!()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;

        // 不支持 self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query"
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            candle_nn::ops::softmax(&attention_scores, D::Minus1)?
        };
        let attention_probs = self.attention_dropout.forward(&attention_probs, false)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(2, 1)?.flatten_from(2)?;

        let projected_context_layer = self.dense.forward(&context_layer)?;
        let projected_context_layer_dropout = self
            .output_dropout
            .forward(&projected_context_layer, false)?;
        let layernormed_context_layer = self
            .layer_norm
            .forward(&(hidden_states + projected_context_layer_dropout)?)?;
        Ok((layernormed_context_layer, attention_probs))
    }
}

// ALBERT_ATTENTION_CLASSES = {
//     "eager": AlbertAttention,
//     "sdpa": AlbertSdpaAttention,
// }

#[derive(Debug, Clone)]
struct AlbertSdpaAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    hidden_size: usize,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,

    attention_dropout: Dropout,
    output_dropout: Dropout,
    dense: Linear,
    layer_norm: LayerNorm,

    span: tracing::Span,
    span_softmax: tracing::Span,

    dropout_prob: f32,
    require_contiguous_qkv: bool,
}

impl AlbertSdpaAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;

        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            query,
            key,
            value,
            hidden_size,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            attention_dropout: Dropout::new(config.attention_probs_dropout_prob),
            output_dropout: Dropout::new(config.hidden_dropout_prob),
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
            dropout_prob: config.attention_probs_dropout_prob,
            require_contiguous_qkv: false,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    // 剪枝
    fn prune(&self, _heads: &[usize]) -> Result<()> {
        unimplemented!()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<(Tensor,)> {
        // batch_size, seq_len, _ = hidden_states.size()
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_output =
            scaled_dot_product_attention(&query_layer, &key_layer, &value_layer, attention_mask)?;
        let attention_output = attention_output.transpose(1, 2)?;
        let attention_output =
            attention_output.reshape((batch_size, seq_len, self.all_head_size))?;

        let projected_context_layer = self.dense.forward(&attention_output)?;
        let projected_context_layer_dropout = self
            .output_dropout
            .forward(&projected_context_layer, false)?;
        let layernormed_context_layer = self
            .layer_norm
            .forward(&(hidden_states + projected_context_layer_dropout)?)?;
        Ok((layernormed_context_layer,))
    }
}

#[derive(Debug, Clone)]
struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}
#[derive(Debug, Clone)]
struct AlbertLayer {
    full_layer_layer_norm: LayerNorm,
    attention: AlbertSdpaAttention,
    ffn: Linear,
    ffn_output: Linear,
    dropout: Dropout,
    activation: HiddenActLayer,
}

impl AlbertLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let full_layer_layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("full_layer_layer_norm"),
        )?;
        let attention = AlbertSdpaAttention::load(vb.pp("attention"), config)?;
        let ffn = linear(config.hidden_size, config.intermediate_size, vb.pp("ffn"))?;
        let ffn_output = linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("ffn_output"),
        )?;
        let activation = HiddenActLayer::new(config.hidden_act);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            // chunk_size_feed_forward: 0,//配置文件没有，transformer 默认是0
            full_layer_layer_norm,
            attention,
            ffn,
            ffn_output,
            dropout,
            activation,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<(Tensor,)> {
        let _enter = self.attention.span.enter();
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;

        // python 版本做了张量分块apply_chunking_to_forward
        // https://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/modeling_albert.py#L290
        // 这里不做分块
        let ffn_output = self.ffn.forward(&attention_output.0)?;
        let ffn_output = self.activation.forward(&ffn_output)?;
        let ffn_output = self.ffn_output.forward(&ffn_output)?;
        // ffn_output + attention_output
        let full_layer = (ffn_output + &attention_output.0)?;
        let hidden_states = self.full_layer_layer_norm.forward(&full_layer)?;

        // (hidden_states,) + attention_output[1:]
        // attention_output[1:] 因为 AlbertSdpaAttention 的原因，什么都没有，
        Ok((hidden_states,))
    }
}

#[derive(Debug, Clone)]
struct AlbertLayers {
    layers: Vec<AlbertLayer>,
}

impl AlbertLayers {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let mut layers = vec![];
        for i in 0..config.inner_group_num {
            layers.push(AlbertLayer::load(vb.pp(i), config)?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<(Tensor,)> {
        let mut hs = hidden_states.clone();
        for layer in self.layers.iter() {
            let layer_output = layer.forward(hidden_states, attention_mask)?;
            hs = layer_output.0;
        }
        Ok((hs,))
    }
}


#[derive(Debug, Clone)]
struct AlbertLayerGroup {
    albert_layers: AlbertLayers,
}

impl AlbertLayerGroup {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let  albert_layers = AlbertLayers::load(vb.pp("albert_layers"), config)?;
        Ok(Self { albert_layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<(Tensor,)> {
        let hs = hidden_states.clone();
        let hs = self.albert_layers.forward(&hs, attention_mask)?.0;
        Ok((hs,))
    }
}

#[derive(Debug, Clone)]
pub struct AlbertLayerGroups {
    groups: Vec<AlbertLayerGroup>,
    num_hidden_layers: usize,
    num_hidden_groups: usize,
}
impl AlbertLayerGroups {
    
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let mut groups = vec![];
        for index in 0..config.num_hidden_groups {
            let group = AlbertLayerGroup::load(vb.pp(index), config)?;
            groups.push(group);
        }
        Ok(Self { groups,num_hidden_layers:config.num_hidden_layers,num_hidden_groups: config.num_hidden_groups })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<(Tensor,)> {
        let mut hidden_states = hidden_states.clone();
        for index in 0..self.num_hidden_layers {
            let layers_per_group = self.num_hidden_layers / self.num_hidden_groups;

            let group_idx = index / layers_per_group;

            let layer_group_output =
                self.groups[group_idx].forward(&hidden_states, attention_mask)?;

            hidden_states = layer_group_output.0;
        }
        Ok((hidden_states,))
    }
}

#[derive(Debug, Clone)]
pub struct AlbertTransformer {
    embedding_hidden_mapping_in: Linear,
    albert_layer_groups: AlbertLayerGroups,
    num_hidden_layers: usize,
    num_hidden_groups: usize,
}

impl AlbertTransformer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embedding_hidden_mapping_in = linear(
            config.embedding_size,
            config.hidden_size,
            vb.pp("embedding_hidden_mapping_in"),
        )?;
        let albert_layer_groups = AlbertLayerGroups::load(vb.pp("albert_layer_groups"), config)?;

        Ok(Self {
            embedding_hidden_mapping_in,
            albert_layer_groups,
            num_hidden_layers: config.num_hidden_layers,
            num_hidden_groups: config.num_hidden_groups,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<(Tensor,)> {
        let hidden_states = self.embedding_hidden_mapping_in.forward(hidden_states)?;
        let hidden_states = self.albert_layer_groups.forward(&hidden_states, attention_mask)?;
        Ok(hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct AlbertModel {
    embeddings: AlbertEmbeddings,
    encoder: AlbertTransformer,
    pooler: Linear,
    // pooler_activation: Activation
    num_hidden_layers: usize,
}

impl AlbertModel {
    pub fn get_input_embeddings(&self) -> &Embedding {
        &self.embeddings.word_embeddings
    }
    pub fn set_input_embeddings(&mut self, new_embeddings: Embedding) {
        self.embeddings.word_embeddings = new_embeddings;
    }

    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embeddings = AlbertEmbeddings::load(vb.pp("embeddings"), config)?;
        let encoder = AlbertTransformer::load(vb.pp("encoder"), config)?;
        let pooler = linear(config.hidden_size, config.hidden_size, vb.pp("pooler"))?;
        let num_hidden_layers = config.num_hidden_layers;
        Ok(Self {
            embeddings,
            encoder,
            pooler,
            num_hidden_layers,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;

        let extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask,
            embedding_output.dtype(),
            Some(seq_len),
        )?;
        //head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        // 因为 head_mask 没有，所以直接用 None: head_mask = [None] * num_hidden_layers
        // 翻译：
        // head_mask = [None] * self.config.num_hidden_layers

        let encoder_outputs = self
            .encoder
            .forward(&embedding_output, &extended_attention_mask)?;
        let sequence_output = encoder_outputs.0;
        //self.pooler_activation(self.pooler(sequence_output[:, 0]))
        let pooled_output = self.pooler.forward(&sequence_output.i((.., 0))?)?;
        let pooled_output = pooled_output.tanh()?;
        //return (sequence_output, pooled_output) + encoder_outputs[1:]
        //encoder_outputs[1:] 为空,不用返回

        Ok((sequence_output, pooled_output))
    }
}

#[derive(Debug, Clone)]
pub struct AlbertForSequenceClassification {
    albert: AlbertModel,
    dropout: Dropout,
    classifier: Linear,
}

impl AlbertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: Config, num_labels: usize) -> Result<Self> {
        let albert = AlbertModel::load(vb.pp("albert"), &config)?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = linear(config.hidden_size, num_labels, vb.pp("classifier"))?;
        Ok(Self {
            albert,
            dropout,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<(Tensor,)> {
        let outputs = self
            .albert
            .forward(input_ids, attention_mask, token_type_ids)?;
        let pooled_output = outputs.1;
        let pooled_output = self.dropout.forward(&pooled_output, false)?;
        let logits = self.classifier.forward(&pooled_output)?;

        Ok((logits,))
    }
}
