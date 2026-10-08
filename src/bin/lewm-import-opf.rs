//! Import an experimental OPF latent model only after cross-framework parity.
use anyhow::ensure;
use candle::{DType, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint::var_builder_from_path,
    models::skyjepa::{SkyJepaModel, checkpoint::SkyJepaCheckpoint, skyjepa_latent_rollout},
    runtime::DeviceSpec,
};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    reference_package: PathBuf,
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    metadata: PathBuf,
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,
}
#[derive(Deserialize)]
struct Fixture {
    batch: usize,
    time: usize,
    states: Vec<f32>,
    actions: Vec<f32>,
    predicted: Vec<f32>,
}
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output_dir.join("checkpoint.json").exists(),
        "output package already exists"
    );
    let reference = SkyJepaCheckpoint::load(&args.reference_package)?;
    let mut contract = reference.contract.clone();
    ensure!(
        contract.model.opf_factors.is_none(),
        "reference must be original SkyJEPA"
    );
    contract.model.opf_factors = Some(4);
    contract.prober = None;
    let device = args.device.resolve()?;
    let vb = var_builder_from_path(&args.weights, DType::F32, &device)?;
    let d = contract.model.latent_dim;
    let basis = vb.get((d, d), "opf_basis")?;
    let gram = basis.matmul(&basis.t()?)?;
    let orth_error = (gram - Tensor::eye(d, DType::F32, &device)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    ensure!(
        orth_error.is_finite() && orth_error < 1e-4,
        "invalid OPF basis: {orth_error}"
    );
    let model = SkyJepaModel::new(contract.model.clone(), vb)?;
    let fixture: Fixture = serde_json::from_slice(&fs::read(&args.fixture)?)?;
    let states = Tensor::from_vec(fixture.states, (fixture.batch, fixture.time, 18), &device)?;
    let actions = Tensor::from_vec(fixture.actions, (fixture.batch, fixture.time, 4), &device)?;
    let actual = skyjepa_latent_rollout(&model, &states, &actions)?.predicted_latents;
    let expected = Tensor::from_vec(
        fixture.predicted,
        (fixture.batch, contract.model.rollout_steps, d),
        &device,
    )?;
    let max_error = (&actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    ensure!(
        max_error.is_finite() && max_error < 1e-3,
        "Python/Candle recursive OPF parity failed: {max_error}"
    );
    let mut provenance: serde_json::Value = serde_json::from_slice(&fs::read(&args.metadata)?)?;
    provenance["import_parity_max_abs"] = serde_json::json!(max_error);
    provenance["import_orthogonality_max_abs"] = serde_json::json!(orth_error);
    provenance["reference_latent_identity"] = serde_json::json!(reference.latent_identity()?);
    let package =
        SkyJepaCheckpoint::publish(&args.output_dir, contract, &args.weights, None, provenance)?;
    println!(
        "{}",
        serde_json::json!({"package":args.output_dir,
        "parity_max_abs":max_error,"orthogonality_max_abs":orth_error,
        "latent_sha256":package.latent_sha256})
    );
    Ok(())
}
