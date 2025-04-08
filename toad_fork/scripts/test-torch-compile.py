import torch
import time

# A more complex function that combines multiple tensor ops:
def complex_tensor_ops(x, W1, b1, W2, b2):
    """
    Perform a sequence of operations:
    1. Linear transform: x * W1 + b1
    2. ReLU
    3. Another linear transform: (previous_result) * W2 + b2
    4. Element-wise multiplication between the output of step 3 and x
    5. A few mathematical ops: sin, cos, log
    6. Return the sum of the final tensor as a scalar
    """
    # Step 1: Linear transform
    out = x @ W1 + b1
    
    # Step 2: ReLU
    out = torch.relu(out)
    
    # Step 3: Another linear transform
    out = out @ W2 + b2
    
    # Step 4: Element-wise multiply with x
    out = out * x
    
    # Step 5: A few random mathematical functions
    # We do sin, cos, log on out; we'll clamp to avoid log(<=0).
    out = torch.sin(out) + torch.cos(out) + torch.log(torch.clamp(out, min=1e-8))
    
    # Step 6: Return the sum (scalar)
    return out.sum()

# Compile the function
compiled_complex_ops = torch.compile(complex_tensor_ops, dynamic=False)

# Let's test on GPU if available, else CPU
assert torch.cuda.is_available()
device = "cuda"
print(f"Using device: {device}")

# Data sizes
batch_size = 2048
input_dim = 1024
hidden_dim = 512

# Create random inputs/weights/biases
x = torch.randn(batch_size, input_dim, device=device)
W1 = torch.randn(input_dim, hidden_dim, device=device)
b1 = torch.randn(hidden_dim, device=device)
W2 = torch.randn(hidden_dim, input_dim, device=device)
b2 = torch.randn(input_dim, device=device)

# Warm-up runs
for _ in range(100):
    complex_tensor_ops(x, W1, b1, W2, b2)
    compiled_complex_ops(x, W1, b1, W2, b2)

# Timing setup
num_iters = 10000

# Original timing
start = time.time()
for _ in range(num_iters):
    _ = complex_tensor_ops(x, W1, b1, W2, b2)
end = time.time()
original_time = end - start

# Compiled timing
start = time.time()
for _ in range(num_iters):
    _ = compiled_complex_ops(x, W1, b1, W2, b2)
end = time.time()
compiled_time = end - start

print(f"Original total time : {original_time:.4f} s")
print(f"Compiled total time : {compiled_time:.4f} s")

speedup = original_time / compiled_time if compiled_time > 0 else float('inf')
print(f"Speedup: {speedup:.2f}x")

