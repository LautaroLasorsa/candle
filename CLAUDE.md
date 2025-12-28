# Candle Development Notes

This document tracks development decisions and implementation notes for the Candle ML framework.

## Mamba Implementation Analysis

### Overview

Candle has **two Mamba implementations** with different design goals:

| Implementation | Path | Purpose | Input Shape |
|---------------|------|---------|-------------|
| `mamba.rs` | `candle-transformers/src/models/mamba.rs` | Incremental inference | `(batch,)` single token |
| `mamba-minimal` | `candle-examples/examples/mamba-minimal/model.rs` | Full sequence processing | `(batch, seq_len)` |

### SSM Background (Selective State Space Model)

Mamba is based on the State Space Model equations from the paper ["Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752):

**Continuous-time SSM:**
```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```

**Discretized SSM (Algorithm 2 in paper):**
```
h_t = Ā h_{t-1} + B̄ x_t
y_t = C h_t + D x_t

where:
  Ā = exp(Δ A)
  B̄ = (Δ A)^{-1} (exp(Δ A) - I) · Δ B  ≈ Δ B  (simplified)
```

**Selective mechanism:** Unlike fixed SSMs, Mamba makes B, C, and Δ input-dependent:
```
B = Linear(x)
C = Linear(x)
Δ = softplus(Linear(x))
```

### Implementation Details

#### 1. `mamba.rs` (Incremental Mode)

**Location:** `candle-transformers/src/models/mamba.rs`

**Design:** Optimized for autoregressive generation (one token at a time).

**Key structures:**
```rust
pub struct State {
    pub hs: Vec<Tensor>,           // Hidden states per layer: (batch, d_inner, D_STATE)
    pub prev_xs: Vec<[Tensor; 4]>, // Conv1d buffer (D_CONV=4 history)
    pub pos: usize,                // Current position (for circular buffer)
}
```

**Forward signature:**
```rust
pub fn forward(&self, input_ids: &Tensor, state: &mut State) -> Result<Tensor>
// input_ids shape: (batch_size,) - single token
```

**SSM computation (single step):**
```rust
// From MambaBlock::forward(), line 149
state.hs[li] = ((&state.hs[li] * (&delta * &a)?.exp()?)? + &delta * &b * &proj_for_conv_b)?;
```

**Performance characteristics:**
- O(n) forward calls for n tokens
- Each call has tensor allocation overhead
- Good for generation (inherently sequential)
- **Poor for prefill** (prompt processing)

#### 2. `mamba-minimal` (Sequence Mode)

**Location:** `candle-examples/examples/mamba-minimal/model.rs`

**Design:** Processes full sequences, based on [mamba-minimal](https://github.com/johnma2006/mamba-minimal).

**Forward signature:**
```rust
fn forward(&self, input_ids: &Tensor) -> Result<Tensor>
// input_ids shape: (batch_size, seq_len)
```

**Selective scan (full sequence):**
```rust
fn selective_scan(u, delta, a, b, c, d) -> Result<Tensor> {
    let (b_sz, l, d_in) = u.dims3()?;
    let mut xs = Tensor::zeros((b_sz, d_in, n), ...)?;
    let mut ys = Vec::with_capacity(l);
    for i in 0..l {  // Still O(n) loop
        xs = ((delta_a.i((.., .., i))? * xs)? + delta_b_u.i((.., .., i))?)?;
        let y = xs.matmul(&c.i((.., i, ..))?.unsqueeze(2)?)?.squeeze(2)?;
        ys.push(y)
    }
    let ys = Tensor::stack(ys.as_slice(), 1)?;
    ys + u.broadcast_mul(d)
}
```

**Performance characteristics:**
- Single forward call for entire sequence
- Better GPU utilization (batched operations)
- Still O(n) sequential loop in selective_scan
- Does NOT maintain state for incremental generation

### Performance Problem

The issue reported: **17+ hours for LLM classification with Candle Mamba**

**Root cause:** Using `mamba.rs` for prompt processing:
```rust
// From candle-examples/examples/mamba/main.rs, line 72-79
for &t in tokens.iter() {
    let input = Tensor::new(&[t], &self.device)?;
    let logits = self.model.forward(&input, &mut state)?;
    // ... N iterations for N tokens
}
```

**Cost breakdown for N=64,000 tokens:**
- 64,000 tensor allocations
- 64,000 kernel launches
- 64,000 state update operations
- ~64-128 seconds just in overhead

### Solution: Unified Implementation

**Goal:** Modify `mamba.rs` to support both modes:

1. **Prefill mode:** Process entire prompt sequence efficiently
2. **Generation mode:** Process single tokens with state (current behavior)

**Proposed API:**
```rust
impl Model {
    /// Process a full sequence (prefill mode) - no state needed
    pub fn forward_prefill(&self, input_ids: &Tensor) -> Result<Tensor> {
        // input_ids: (batch, seq_len)
        // Uses selective_scan for efficient sequence processing
    }

    /// Process single token with state (generation mode)
    pub fn forward(&self, input_ids: &Tensor, state: &mut State) -> Result<Tensor> {
        // input_ids: (batch,) - single token
        // Current implementation
    }

    /// Initialize state from prefill output (for switching modes)
    pub fn init_state_from_prefill(&self, ...) -> Result<State> {
        // Extract final state from prefill for continued generation
    }
}
```

### Parallel Scan Algorithm (Future Optimization)

The current selective_scan has O(n) sequential depth. The paper describes a parallel scan approach:

**Associative operation for SSM:**
```
(h_i, y_i) ⊕ (h_j, y_j) = (Ā_j h_i + h_j, y_i ∥ y_j)
```

**Parallel scan achieves O(log n) depth:**
```
Step 1: Process pairs (0,1), (2,3), (4,5), ...
Step 2: Combine results in tree structure
Step k: Final result after log₂(n) steps
```

This requires custom CUDA kernels for efficient implementation. The official Mamba uses:
- `selective_scan_cuda.cu` for the parallel kernel
- Memory-efficient recomputation in backward pass

**For this PR:** Focus on the unified API first. Parallel scan can be a follow-up optimization.

---

## Implementation: `mamba_prefill.rs`

### Location

`candle-transformers/src/models/mamba_prefill.rs`

### Overview

New alternative implementation that supports efficient sequence (prefill) processing while maintaining compatibility with generation mode. Created as a separate file to avoid breaking existing code using `mamba.rs`.

### Key Differences from `mamba.rs`

| Feature | `mamba.rs` | `mamba_prefill.rs` |
|---------|------------|-------------------|
| Input shape | `(batch,)` single token | `(batch, seq_len)` or `(batch,)` |
| Conv1d | Manual circular buffer | `candle_nn::Conv1d` with causal padding |
| State | External, passed by reference | Returned after prefill |
| API | `forward(token, &mut state)` | `forward_prefill(seq)` + `forward_generate(token, &mut state)` |

### API

```rust
impl Model {
    /// Process full sequence, returns logits + state for generation
    pub fn forward_prefill(&self, input_ids: &Tensor) -> Result<(Tensor, State)>
    // input_ids: (batch, seq_len)
    // Returns: (logits for last token, state for continued generation)

    /// Generate single token using state from prefill
    pub fn forward_generate(&self, input_id: &Tensor, state: &mut State) -> Result<Tensor>
    // input_id: (batch,) single token
    // Returns: logits

    /// Process full sequence, return all logits (implements Module trait)
    pub fn forward_all(&self, input_ids: &Tensor) -> Result<Tensor>
    // input_ids: (batch, seq_len)
    // Returns: (batch, seq_len, vocab_size)
}
```

### Usage Example

```rust
use candle_transformers::models::mamba_prefill::{Model, Config, State};

// Load model
let model = Model::new(&config, vb)?;

// Prefill: process entire prompt efficiently
let prompt = Tensor::new(&[1u32, 2, 3, 4, 5], &device)?.unsqueeze(0)?;
let (logits, mut state) = model.forward_prefill(&prompt)?;

// Sample next token from logits
let next_token = sample(&logits)?;

// Generate: continue token by token
for _ in 0..max_tokens {
    let input = Tensor::new(&[next_token], &device)?;
    let logits = model.forward_generate(&input, &mut state)?;
    next_token = sample(&logits)?;
}
```

### Implementation Details

#### Selective Scan (`selective_scan`)

Implements the discretized SSM equations over a full sequence:

```
h_t = exp(Δ_t * A) * h_{t-1} + Δ_t * B_t * x_t
y_t = C_t * h_t + D * x_t
```

- Input: `u` (batch, seq_len, d_inner), `delta`, `a`, `b`, `c`, `d`
- Output: `y` (batch, seq_len, d_inner), `final_h` (batch, d_inner, d_state)
- Complexity: O(seq_len) sequential loop (parallel scan is future work)

#### State Management

```rust
pub struct State {
    pub hs: Vec<Tensor>,        // SSM hidden states per layer
    pub conv_states: Vec<Tensor>, // Conv history per layer
    pub pos: usize,             // Position counter
}
```

- `hs`: Shape `(batch, d_inner, d_state)` per layer
- `conv_states`: Shape `(batch, d_inner, d_conv)` per layer (d_conv=4)
- Initialized after prefill, updated during generation

#### Conv1d Handling

- Uses `candle_nn::Conv1d` with depthwise convolution (`groups=d_inner`)
- Causal padding: `padding=d_conv-1`, then narrow to `seq_len`
- Generation mode: shift-and-append for single-token processing

### Tests

Basic tests included:
- `test_config`: Verify config calculations
- `test_state_creation`: Verify state initialization

### Future Work

- Parallel scan algorithm for O(log n) depth
- CUDA kernel for maximum performance
- Benchmarks comparing prefill vs token-by-token

---

## Implementation Plan

### Phase 1: Unified API ✅ COMPLETED

1. ✅ Created `mamba_prefill.rs` as alternative implementation
2. ✅ Implemented `selective_scan()` for sequence processing
3. ✅ Implemented `forward_prefill()` returning state for generation
4. ✅ Implemented `forward_generate()` for token-by-token generation
5. ✅ Added module export in `mod.rs`
6. ✅ Added documentation and basic tests

### Phase 2: Parallel Scan (Future)

1. Implement work-efficient parallel scan in pure Rust/Candle
2. Add optional CUDA kernel for maximum performance
3. Benchmark against sequential implementation

---

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752) - Original paper
- [mamba-minimal](https://github.com/johnma2006/mamba-minimal) - Simple Python implementation
- [Official Mamba](https://github.com/state-spaces/mamba) - Reference with CUDA kernels
- [Blelloch 1990](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) - Parallel Prefix algorithms
