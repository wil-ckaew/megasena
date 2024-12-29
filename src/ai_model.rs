//src/ai_model.rs
use tch::{nn, Device};

pub fn create_model(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(vs / "fc1", 6, 256, Default::default()))  // Ajustado para 6 entradas
        .add_fn(|xs| xs.relu().dropout(0.3, true))
        .add(nn::linear(vs / "fc2", 256, 512, Default::default()))
        .add_fn(|xs| xs.relu().dropout(0.3, true))
        .add(nn::linear(vs / "fc3", 512, 256, Default::default()))
        .add_fn(|xs| xs.relu().dropout(0.3, true))
        .add(nn::linear(vs / "fc4", 256, 6, Default::default()))  // Saída com 6 números
}
