// src/main.rs
use std::error::Error;
use std::fs::OpenOptions;
use std::io::{self, Write};
use csv::{ReaderBuilder, WriterBuilder};
use tch::{nn, Device, Tensor, Kind, Reduction};
use tch::nn::OptimizerConfig;
use tch::manual_seed;
use crate::ai_model::create_model;
use crate::data_loaders::load_data_from_csv;  // Mantém apenas a importação

mod ai_model;
mod data_loaders;

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "resultado_megasena.csv"; // Alterado para refletir a Mega-Sena
    let data = load_data_from_csv(file_path)?;  // Chama diretamente do módulo data_loaders
    let data_tensor = Tensor::of_slice(&data.iter().flatten().copied().collect::<Vec<f32>>())
        .view([-1, 6]); // As 6 bolas de cada sorteio da Mega-Sena

    let targets = data_tensor.shallow_clone(); // Usando os próprios dados como alvo (para simplificação)

    let mut generated_numbers: Vec<Vec<i32>> = Vec::new();

    for i in 1..=3 {
        manual_seed(i as i64);
        let vs = nn::VarStore::new(Device::Cpu);
        let model = create_model(&vs.root());
        let mut optimizer = nn::Adam::default().build(&vs, 1e-4)?;
        train_model(&model, &data_tensor, &targets, &mut optimizer, 10000, i);
        vs.save(&format!("model_megasena_{}.pth", i))?;
        println!("Modelo {} treinado e salvo com sucesso!", i);
        let result = generate_predictions(&model, &data_tensor, i);
        generated_numbers.push(result);
    }

    let drawn_result = get_drawn_result_from_user()?;
    append_to_csv(&[drawn_result.clone()], file_path)?;

    for (i, result) in generated_numbers.iter().enumerate() {
        let acertos = compare_results(result, &drawn_result);
        println!("Modelo {}: {} acertos", i + 1, acertos);
    }

    Ok(())
}

fn get_drawn_result_from_user() -> Result<Vec<i32>, Box<dyn Error>> {
    println!("Digite os 6 números sorteados (separados por espaço):");
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let drawn_result: Vec<i32> = input
        .trim()
        .split_whitespace()
        .filter_map(|s| s.parse::<i32>().ok())
        .collect();

    if drawn_result.len() != 6 {  // Ajustado para 6 números
        return Err("Você precisa inserir exatamente 6 números.".into());
    }

    Ok(drawn_result)
}

fn append_to_csv(generated_numbers: &[Vec<i32>], file_path: &str) -> Result<(), Box<dyn Error>> {
    let file = OpenOptions::new().append(true).create(true).open(file_path)?;
    let mut wtr = WriterBuilder::new().has_headers(false).from_writer(file);
    for numbers in generated_numbers {
        wtr.write_record(numbers.iter().map(|&num| num.to_string()))?;
    }
    wtr.flush()?;
    Ok(())
}

fn compare_results(prediction: &[i32], drawn_result: &[i32]) -> usize {
    prediction.iter().filter(|&&x| drawn_result.contains(&x)).count()
}

fn train_model(
    model: &impl nn::Module, 
    data_tensor: &Tensor, 
    targets: &Tensor, 
    optimizer: &mut nn::Optimizer, 
    epochs: usize, 
    model_id: usize
) {
    for epoch in 1..=epochs {
        optimizer.zero_grad();
        let outputs = model.forward(data_tensor);
        let loss = outputs.mse_loss(targets, Reduction::Mean);
        loss.backward();
        optimizer.step();

        if epoch % 1000 == 0 {
            println!(
                "Modelo {} - Epoch {}: Loss = {:.6}",
                model_id, epoch, loss.double_value(&[])
            );
        }
    }
}

fn generate_predictions(model: &impl nn::Module, data_tensor: &Tensor, model_id: usize) -> Vec<i32> {
    let predictions = model.forward(data_tensor);
    let last_prediction = predictions.get(predictions.size()[0] - 1);
    let f32_vec: Vec<f32> = last_prediction.into();
    let result: Vec<i32> = f32_vec.into_iter().map(|value| value.round().clamp(1.0, 60.0) as i32).collect(); // Ajustado para a Mega-Sena com números de 1 a 60
    println!("Previsão dos próximos 6 números (Modelo {}): {:?}", model_id, result);
    result
}
