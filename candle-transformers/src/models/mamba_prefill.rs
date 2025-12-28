//! Mamba implementation with efficient sequence (prefill) processing.
//!
//! This is an alternative implementation to `mamba.rs` that supports processing
//! entire sequences efficiently, making it suitable for prompt processing (prefill).
//!
//! See ["Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752)
//!
//! ## Comparison with `mamba.rs`
//!
//! | Feature | `mamba.rs` | `mamba_prefill.rs` (this) |
//! |---------|------------|---------------------------|
//! | Input shape | `(batch,)` single token | `(batch, seq_len)` sequence |
//! | State management | External `State` struct | Internal, returned after prefill |
//! | Use case | Token-by-token generation | Prompt processing + generation |
//! | Performance | O(n) forward calls | O(1) forward call for prefill |
//!
//! ## Usage
//!
//! ```ignore
//! use candle_transformers::models::mamba_prefill::{Model, Config};
//!
//! // Load model
//! let model = Model::new(&config, vb)?;
//!
//! // Prefill: process entire prompt in one call
//! let prompt_tokens = Tensor::new(&[1u32, 2, 3, 4, 5], &device)?.unsqueeze(0)?;
//! let (logits, state) = model.forward_prefill(&prompt_tokens)?;
//!
//! // Generation: continue token by token using the state
//! let mut state = state;
//! for _ in 0..max_new_tokens {
//!     let next_token = sample(&logits);
//!     let input = Tensor::new(&[next_token], &device)?;
//!     let logits = model.forward_generate(&input, &mut state)?;
//! }
//! ```
//!
//! Based on [mamba-minimal](https://github.com/johnma2006/mamba-minimal)

use crate::models::with_tracing::{linear, linear_no_bias, Linear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

const D_CONV: usize = 4;
const D_STATE: usize = 16;

// =============================================================================
// Configuration
// =============================================================================

/// Model configuration. Compatible with `mamba.rs` Config.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    pub pad_vocab_size_multiple: usize,
}

impl Config {
    fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple;
        self.vocab_size.div_ceil(pad) * pad
    }

    fn dt_rank(&self) -> usize {
        self.d_model.div_ceil(16)
    }

    fn d_inner(&self) -> usize {
        self.d_model * 2
    }
}

// =============================================================================
// State for Generation Mode
// =============================================================================

/// State for incremental (token-by-token) generation after prefill.
///
/// This state is returned by `forward_prefill()` and can be used with
/// `forward_generate()` for continued generation.
pub struct State {
    /// Hidden states for each layer: (batch, d_inner, d_state)
    pub hs: Vec<Tensor>,
    /// Convolution history buffer for each layer
    pub conv_states: Vec<Tensor>,
    /// Current position in sequence
    pub pos: usize,
}

impl State {
    /// Creates a new zero-initialized state.
    pub fn new(batch_size: usize, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let mut hs = Vec::with_capacity(cfg.n_layer);
        let mut conv_states = Vec::with_capacity(cfg.n_layer);

        for _ in 0..cfg.n_layer {
            hs.push(Tensor::zeros((batch_size, d_inner, D_STATE), dtype, device)?);
            conv_states.push(Tensor::zeros((batch_size, d_inner, D_CONV), dtype, device)?);
        }

        Ok(Self {
            hs,
            conv_states,
            pos: 0,
        })
    }
}

// =============================================================================
// Selective Scan (Core SSM Operation)
// =============================================================================

/// Performs selective scan over a full sequence.
///
/// Implements the discretized SSM:
/// ```text
/// h_t = exp(Δ_t * A) * h_{t-1} + Δ_t * B_t * x_t
/// y_t = C_t * h_t + D * x_t
/// ```
///
/// # Arguments
/// * `u` - Input tensor: (batch, seq_len, d_inner)
/// * `delta` - Time step: (batch, seq_len, d_inner)
/// * `a` - State matrix A: (d_inner, d_state)
/// * `b` - Input projection B: (batch, seq_len, d_state)
/// * `c` - Output projection C: (batch, seq_len, d_state)
/// * `d` - Skip connection D: (d_inner,)
///
/// # Returns
/// * Output: (batch, seq_len, d_inner)
/// * Final hidden state: (batch, d_inner, d_state)
fn selective_scan(
    u: &Tensor,
    delta: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b_sz, seq_len, d_in) = u.dims3()?;
    let n = a.dim(1)?; // d_state

    // Compute discretization terms
    // delta_a = exp(Δ * A)
    let delta_t = delta.transpose(1, 2)?; // (b, d, l)
    let delta_expanded = delta_t.unsqueeze(D::Minus1)?; // (b, d, l, 1)
    let a_expanded = a.reshape((1, d_in, 1, n))?; // (1, d, 1, n)
    let delta_a = delta_expanded.broadcast_mul(&a_expanded)?.exp()?; // (b, d, l, n)

    // delta_b_u = Δ * B * u
    let b_t = b.transpose(1, 2)?; // (b, n, l)
    let b_expanded = b_t.unsqueeze(1)?; // (b, 1, n, l)
    let u_t = u.transpose(1, 2)?; // (b, d, l)
    let u_expanded = u_t.unsqueeze(D::Minus1)?; // (b, d, l, 1)

    let delta_b = delta_expanded.broadcast_mul(&b_expanded.transpose(2, 3)?)?; // (b, d, l, n)
    let delta_b_u = delta_b.broadcast_mul(&u_expanded)?; // (b, d, l, n)

    // Sequential scan (TODO: replace with parallel scan for O(log n))
    let dtype = delta_a.dtype();
    let device = delta_a.device();
    let mut h = Tensor::zeros((b_sz, d_in, n), dtype, device)?;
    let mut ys = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        // h_t = delta_a[t] * h_{t-1} + delta_b_u[t]
        let delta_a_t = delta_a.i((.., .., t, ..))?.squeeze(2)?; // (b, d, n)
        let delta_b_u_t = delta_b_u.i((.., .., t, ..))?.squeeze(2)?; // (b, d, n)
        h = (delta_a_t.broadcast_mul(&h)? + delta_b_u_t)?;

        // y_t = h_t @ c_t
        let c_t = c.i((.., t, ..))?.unsqueeze(D::Minus1)?; // (b, n, 1)
        let y_t = h.matmul(&c_t)?.squeeze(D::Minus1)?; // (b, d)
        ys.push(y_t);
    }

    // Stack outputs and add skip connection
    let y = Tensor::stack(&ys, 1)?; // (b, l, d)
    let d_expanded = d.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, d)
    let output = (y + u.broadcast_mul(&d_expanded)?)?;

    Ok((output, h))
}

// =============================================================================
// Mamba Block
// =============================================================================

/// A single Mamba block with sequence processing support.
#[derive(Clone, Debug)]
pub struct MambaBlock {
    in_proj: Linear,
    conv1d: candle_nn::Conv1d,
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
    dt_rank: usize,
    d_inner: usize,
}

impl MambaBlock {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let dt_rank = cfg.dt_rank();

        let in_proj = linear_no_bias(cfg.d_model, d_inner * 2, vb.pp("in_proj"))?;

        // Conv1d with groups=d_inner (depthwise)
        let conv_cfg = candle_nn::Conv1dConfig {
            groups: d_inner,
            padding: D_CONV - 1,
            ..Default::default()
        };
        let conv1d = candle_nn::conv1d(d_inner, d_inner, D_CONV, conv_cfg, vb.pp("conv1d"))?;

        let x_proj = linear_no_bias(d_inner, dt_rank + D_STATE * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;
        let a_log = vb.get((d_inner, D_STATE), "A_log")?;
        let d = vb.get(d_inner, "D")?;
        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            conv1d,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            dt_rank,
            d_inner,
        })
    }

    /// Forward pass for sequence processing (prefill mode).
    ///
    /// # Arguments
    /// * `xs` - Input tensor: (batch, seq_len, d_model)
    ///
    /// # Returns
    /// * Output: (batch, seq_len, d_model)
    /// * Final SSM state: (batch, d_inner, d_state)
    /// * Final conv state: (batch, d_inner, d_conv)
    pub fn forward_sequence(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _d_model) = xs.dims3()?;

        // Project input and split into x and residual
        let xs_proj = xs.apply(&self.in_proj)?; // (b, l, 2*d_inner)
        let chunks = xs_proj.chunk(2, D::Minus1)?;
        let x = &chunks[0]; // (b, l, d_inner)
        let res = &chunks[1]; // (b, l, d_inner)

        // Apply causal conv1d
        let x_conv = x.transpose(1, 2)?; // (b, d_inner, l)
        let x_conv = self.conv1d.forward(&x_conv)?; // (b, d_inner, l + padding)
        let x_conv = x_conv.narrow(2, 0, seq_len)?; // (b, d_inner, l) - causal
        let x_conv = x_conv.transpose(1, 2)?; // (b, l, d_inner)
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        // Extract conv state (last D_CONV positions for generation)
        let conv_state = if seq_len >= D_CONV {
            x.transpose(1, 2)?.narrow(2, seq_len - D_CONV, D_CONV)?
        } else {
            // Pad if sequence is shorter than D_CONV
            let padding = Tensor::zeros(
                (b_sz, self.d_inner, D_CONV - seq_len),
                x.dtype(),
                x.device(),
            )?;
            Tensor::cat(&[&padding, &x.transpose(1, 2)?], 2)?
        };

        // SSM projections
        let x_dbl = x_conv.apply(&self.x_proj)?; // (b, l, dt_rank + 2*d_state)
        let delta = x_dbl.narrow(D::Minus1, 0, self.dt_rank)?;
        let b = x_dbl.narrow(D::Minus1, self.dt_rank, D_STATE)?;
        let c = x_dbl.narrow(D::Minus1, self.dt_rank + D_STATE, D_STATE)?;

        // Project delta and apply softplus
        let delta = delta.contiguous()?.apply(&self.dt_proj)?;
        let delta = (delta.exp()? + 1.0)?.log()?; // softplus

        // Compute A (negative exp of learned log)
        let a = self.a_log.to_dtype(delta.dtype())?.exp()?.neg()?;

        // Run selective scan
        let (ssm_out, final_h) = selective_scan(&x_conv, &delta, &a, &b, &c, &self.d)?;

        // Apply gating and output projection
        let y = (ssm_out * candle_nn::ops::silu(res)?)?;
        let output = y.apply(&self.out_proj)?;

        Ok((output, final_h, conv_state))
    }

    /// Forward pass for single token generation.
    ///
    /// # Arguments
    /// * `x` - Input tensor: (batch, d_model)
    /// * `h` - SSM hidden state: (batch, d_inner, d_state)
    /// * `conv_state` - Conv history: (batch, d_inner, d_conv)
    ///
    /// # Returns
    /// * Output: (batch, d_model)
    /// * Updated h
    /// * Updated conv_state
    pub fn forward_step(
        &self,
        x: &Tensor,
        h: &Tensor,
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (_b_sz, _d_model) = x.dims2()?;

        // Project and split
        let x_proj = x.apply(&self.in_proj)?; // (b, 2*d_inner)
        let chunks = x_proj.chunk(2, D::Minus1)?;
        let x_in = &chunks[0]; // (b, d_inner)
        let res = &chunks[1]; // (b, d_inner)

        // Update conv state (shift left, append new)
        let new_conv_state = Tensor::cat(
            &[
                &conv_state.narrow(2, 1, D_CONV - 1)?,
                &x_in.unsqueeze(D::Minus1)?,
            ],
            2,
        )?;

        // Apply conv (sum over history with weights)
        let conv_weight = self.conv1d.weight().squeeze(1)?; // (d_inner, d_conv)
        let x_conv = new_conv_state
            .broadcast_mul(&conv_weight.unsqueeze(0)?)?
            .sum(D::Minus1)?;
        // Bias is (d_inner,), x_conv is (b, d_inner) - need to unsqueeze for broadcast
        let bias = self.conv1d.bias().unwrap().unsqueeze(0)?;
        let x_conv = x_conv.broadcast_add(&bias)?;
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        // SSM projections
        let x_dbl = x_conv.apply(&self.x_proj)?;
        let delta = x_dbl.narrow(D::Minus1, 0, self.dt_rank)?.contiguous()?;
        let b = x_dbl.narrow(D::Minus1, self.dt_rank, D_STATE)?;
        let c = x_dbl.narrow(D::Minus1, self.dt_rank + D_STATE, D_STATE)?;

        let delta = delta.apply(&self.dt_proj)?;
        let delta = (delta.exp()? + 1.0)?.log()?;

        let a = self.a_log.to_dtype(delta.dtype())?.exp()?.neg()?;
        let d = self.d.to_dtype(delta.dtype())?;

        // Single-step SSM update
        // h_new = exp(delta * A) * h + delta * B * x
        let delta_exp = delta.unsqueeze(D::Minus1)?; // (b, d_inner, 1)
        let a_exp = a.unsqueeze(0)?; // (1, d_inner, d_state)
        let delta_a = delta_exp.broadcast_mul(&a_exp)?.exp()?;

        let b_exp = b.unsqueeze(1)?; // (b, 1, d_state)
        let x_exp = x_conv.unsqueeze(D::Minus1)?; // (b, d_inner, 1)
        let delta_b_x = delta_exp.broadcast_mul(&b_exp)?.broadcast_mul(&x_exp)?;

        let h_new = (delta_a.broadcast_mul(h)? + delta_b_x)?;

        // Output: y = h @ c + d * x
        let c_exp = c.unsqueeze(D::Minus1)?; // (b, d_state, 1)
        let y = h_new.matmul(&c_exp)?.squeeze(D::Minus1)?; // (b, d_inner)
        // d is (d_inner,), need to unsqueeze for broadcast with (b, d_inner)
        let d_exp = d.unsqueeze(0)?;
        let y = y.broadcast_add(&x_conv.broadcast_mul(&d_exp)?)?;

        // Gate and project
        let out = (y * candle_nn::ops::silu(res)?)?;
        let out = out.apply(&self.out_proj)?;

        Ok((out, h_new, new_conv_state))
    }
}

// =============================================================================
// Residual Block
// =============================================================================

#[derive(Clone, Debug)]
pub struct ResidualBlock {
    mixer: MambaBlock,
    norm: RmsNorm,
}

impl ResidualBlock {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let mixer = MambaBlock::new(cfg, vb.pp("mixer"))?;
        Ok(Self { mixer, norm })
    }

    pub fn forward_sequence(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = xs.apply(&self.norm)?;
        let (out, h, conv_state) = self.mixer.forward_sequence(&normed)?;
        let output = (out + xs)?;
        Ok((output, h, conv_state))
    }

    pub fn forward_step(
        &self,
        x: &Tensor,
        h: &Tensor,
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = x.apply(&self.norm)?;
        let (out, h_new, conv_new) = self.mixer.forward_step(&normed, h, conv_state)?;
        let output = (out + x)?;
        Ok((output, h_new, conv_new))
    }
}

// =============================================================================
// Model
// =============================================================================

/// Mamba model with prefill and generation support.
#[derive(Clone, Debug)]
pub struct Model {
    embedding: candle_nn::Embedding,
    layers: Vec<ResidualBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
    dtype: DType,
    n_layer: usize,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size(), cfg.d_model, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.n_layer {
            let layer = ResidualBlock::new(cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm_f = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
        let lm_head = Linear::from_weights(embedding.embeddings().clone(), None);

        Ok(Self {
            embedding,
            layers,
            norm_f,
            lm_head,
            dtype: vb.dtype(),
            n_layer: cfg.n_layer,
        })
    }

    /// Process an entire sequence (prefill mode).
    ///
    /// This is efficient for processing prompts - single forward pass
    /// instead of token-by-token.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs: (batch, seq_len)
    ///
    /// # Returns
    /// * Logits for the last token: (batch, vocab_size)
    /// * State for continued generation
    pub fn forward_prefill(&self, input_ids: &Tensor) -> Result<(Tensor, State)> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut xs = self.embedding.forward(input_ids)?; // (b, l, d_model)

        // Collect states from each layer
        let mut hs = Vec::with_capacity(self.n_layer);
        let mut conv_states = Vec::with_capacity(self.n_layer);

        // Process through layers
        for layer in &self.layers {
            let (out, h, conv_state) = layer.forward_sequence(&xs)?;
            xs = out;
            hs.push(h);
            conv_states.push(conv_state);
        }

        // Get logits for last position only
        let last_hidden = xs.narrow(1, seq_len - 1, 1)?.squeeze(1)?; // (b, d_model)
        let logits = last_hidden.apply(&self.norm_f)?.apply(&self.lm_head)?;

        let state = State {
            hs,
            conv_states,
            pos: seq_len,
        };

        Ok((logits, state))
    }

    /// Generate a single token given previous state.
    ///
    /// Use this after `forward_prefill()` for continued generation.
    ///
    /// # Arguments
    /// * `input_id` - Single token ID: (batch,)
    /// * `state` - State from prefill or previous generation step
    ///
    /// # Returns
    /// * Logits: (batch, vocab_size)
    pub fn forward_generate(&self, input_id: &Tensor, state: &mut State) -> Result<Tensor> {
        let _b_sz = input_id.dims1()?;

        // Embed single token
        let mut x = self.embedding.forward(input_id)?; // (b, d_model)

        // Process through layers
        for (i, layer) in self.layers.iter().enumerate() {
            let (out, h_new, conv_new) = layer.forward_step(&x, &state.hs[i], &state.conv_states[i])?;
            x = out;
            state.hs[i] = h_new;
            state.conv_states[i] = conv_new;
        }

        state.pos += 1;

        // Get logits
        x.apply(&self.norm_f)?.apply(&self.lm_head)
    }

    /// Convenience method: process full sequence and return all logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs: (batch, seq_len)
    ///
    /// # Returns
    /// * Logits for all positions: (batch, seq_len, vocab_size)
    pub fn forward_all(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _dims = input_ids.dims2()?;

        let mut xs = self.embedding.forward(input_ids)?;

        for layer in &self.layers {
            let (out, _, _) = layer.forward_sequence(&xs)?;
            xs = out;
        }

        // Apply final norm and lm_head to all positions
        xs.apply(&self.norm_f)?.apply(&self.lm_head)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

// =============================================================================
// Module implementation for compatibility
// =============================================================================

impl Module for Model {
    /// Forward pass processing full sequence.
    /// Returns logits for ALL positions: (batch, seq_len, vocab_size)
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_all(input_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn test_config() {
        let cfg = Config {
            d_model: 768,
            n_layer: 24,
            vocab_size: 50257,
            pad_vocab_size_multiple: 8,
        };
        assert_eq!(cfg.vocab_size(), 50264); // Padded
        assert_eq!(cfg.d_inner(), 1536);
        assert_eq!(cfg.dt_rank(), 48);
    }

    #[test]
    fn test_state_creation() -> Result<()> {
        let cfg = Config {
            d_model: 64,
            n_layer: 2,
            vocab_size: 100,
            pad_vocab_size_multiple: 1,
        };
        let state = State::new(1, &cfg, DType::F32, &Device::Cpu)?;
        assert_eq!(state.hs.len(), 2);
        assert_eq!(state.conv_states.len(), 2);
        assert_eq!(state.pos, 0);
        Ok(())
    }
}
