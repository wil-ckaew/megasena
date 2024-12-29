// src/data_loaders.rs
use std::error::Error;
use csv::ReaderBuilder;

pub fn load_data_from_csv(file_path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(file_path)?;
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let row: Vec<f32> = record.iter()
            .skip(2)  // Ignora as duas primeiras colunas (Concurso e Data)
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();
        if row.len() == 6 {  // Ajustado para 6 bolas
            data.push(row);
        }
    }
    Ok(data)
}
