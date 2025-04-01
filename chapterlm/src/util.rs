


use std::collections::HashSet;

use candle_core::{ DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{ops, Linear, Module};

/// 剪枝 linear 层
///
/// # Arguments
/// * `layer` - The linear layer to prune.
/// * `index` - The indices to keep (1D tensor).
/// * `dim` - The dimension to prune (0 for output, 1 for input).
///
/// # Returns
/// A new pruned `Linear` layer.
pub fn prune_linear_layer(
    layer: &Linear,
    index: &Tensor,
    dim: usize,
) -> Result<Linear> {
    let device = layer.weight().device();
    let index = index.to_device(device)?;

    // Prune the weight matrix
    let pruned_weight = layer.weight().index_select(&index, dim)?;

    // Prune the bias (if exists)
    let pruned_bias = match layer.bias() {
        Some(bias) => {
            if dim == 1 {
                // If pruning input dim, bias remains unchanged
                Some(bias.clone())
            } else {
                // If pruning output dim, index-select the bias
                Some(bias.index_select(&index, 0)?)
            }
        }
        None => None,
    };

    // Directly create a new linear layer from tensors
    Ok(Linear::new(pruned_weight, pruned_bias))
}

/// attn_mask: Tensor[[1, 1, 64, 64], f32]
pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor,attn_mask: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    // L, S = query.size(-2), key.size(-2)
    let l = q.dim(D::Minus2)?;
    let s = k.dim(D::Minus2)?;

    let attn_bias = Tensor::zeros((l,s), q.dtype(), q.device())?; //Tensor[[64, 64], f32]
    println!("attn_mask::{attn_mask}");
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    // let attn_bias = (attn_mask + attn_bias)?;
    let attn_bias = match attn_mask.dtype() {
        DType::U8 => attn_mask.where_cond(
            &attn_bias,
            &Tensor::new(f32::NEG_INFINITY, q.device())?
        )?,
        _ => (attn_mask.broadcast_add(&attn_bias))?,
    };
    let attn_weights = (attn_weights.broadcast_add(&attn_bias))?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}


pub fn _prepare_4d_attention_mask_for_sdpa(mask:&Tensor,dtype:DType,tgt_len:Option<usize>)->candle_core::Result<Tensor> {
    //_, key_value_length = mask.shape
    let (bsz, key_value_length) = mask.dims2()?;
    let tgt_len = match tgt_len {
        Some(tgt_len) => tgt_len,
        None => key_value_length,
    };

    //  expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    let expanded_mask = mask.unsqueeze(1)?.unsqueeze(1)?.expand((bsz,1,tgt_len,key_value_length))?;
    // 变更类型，u32 转f32
    let expanded_mask = expanded_mask.to_dtype(DType::F32)?;
    println!("expanded_mask::{expanded_mask}");
    //inverted_mask = 1.0 - expanded_mask
    let inverted_mask = Tensor::ones((bsz,1,tgt_len,key_value_length),dtype,mask.device())? - &expanded_mask;
    //inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    let inverted_mask = inverted_mask?;
    let mask = inverted_mask.ne(0.0f32)?;
    masked_fill( &expanded_mask,&mask, f32::NEG_INFINITY)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

