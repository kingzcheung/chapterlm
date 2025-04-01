use std::path::Path;

use candle_core::Tensor;
use candle_nn::VarBuilder;
use chapterlm::{albert::AlbertForSequenceClassification, config::Config};
use tokenizers::Tokenizer;

fn main() {
    // 获取项目根目录
    let root_dir = env!("CARGO_MANIFEST_DIR");

    let config_filename = Path::new(root_dir).join("../rkingzhong/chapterlm/config.json");
    let tokenizer_filename = Path::new(root_dir).join("../rkingzhong/chapterlm/tokenizer.json");
    let weights_filename = Path::new(root_dir).join("../rkingzhong/chapterlm/model.safetensors");
    let config = std::fs::read_to_string(config_filename).unwrap();
    let config: Config = serde_json::from_str(&config).unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
    let device = candle_core::Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename], candle_core::DType::F32, &device)
            .unwrap()
    };

    let model = AlbertForSequenceClassification::load(vb, config, 2).unwrap();

    let tokens = tokenizer.encode("1、回归", true).unwrap();

    let token_ids = {
        let tokens = tokens.get_ids().to_vec();
        Tensor::new(tokens.as_slice(), &candle_core::Device::Cpu).unwrap()
    }.unsqueeze(0).unwrap();

    // println!("token_ids::{token_ids}");

    let attention_mask = {
        let tokens = tokens.get_attention_mask().to_vec();
        Tensor::new(tokens.as_slice(), &candle_core::Device::Cpu).unwrap().unsqueeze(0).unwrap()
    };
    // println!("attention_mask::{attention_mask}");


    let token_type_ids = token_ids.zeros_like().unwrap();

    // println!("token_type_ids::{token_type_ids}");


    let ys = model
        .forward(&token_ids, &attention_mask, &token_type_ids)
        .unwrap();

    // softmax
    //print("是章节名" if pred == 1 else "不是章节名")
    let pred = ys.0.argmax(1).unwrap();
    println!("{}",pred);
}
