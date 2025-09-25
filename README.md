## How to Obtain Key Embeddings

### Step-by-Step Guide

1. **Locate the Model Architecture Source Code**  
   Find the corresponding source code file for the large model architecture. For example, for LLaMA2, you need to locate the `modeling_llama.py` file.

2. **Modify the Attention Class**  
   In the `LlamaAttention` class (e.g., `class LlamaAttention(nn.Module)`), modify the return value to include `key_states`:

   `attn_output = attn_output.reshape(*input_shape, -1).contiguous()`
  
   `attn_output = self.o_proj(attn_output)`
  
   `return attn_output, attn_weights, key_states # Add key_states to the return tuple`

   **Note:** Ensure corresponding calls to this class are updated accordingly.

3. **Use Hooks to Extract Key States**  
The provided code uses hooks to capture the `key_states` efficiently.

---

### Code Availability Statement

We provide the code necessary for researching SAT properties. For the ProtoKV experiment code on LongBench and Ruler, the modules are relatively extensive and are currently being organized. We commit to open-sourcing this code upon the acceptance of our associated paper.
