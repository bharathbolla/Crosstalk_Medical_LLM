"""
Find optimal batch size for your GPU automatically.
Run this BEFORE your experiments to maximize efficiency!
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import gc


def test_batch_size(model_name, max_length=512, start_batch=8, max_batch=64):
    """
    Test different batch sizes to find the maximum that fits in VRAM.

    Args:
        model_name: HuggingFace model name
        max_length: Maximum sequence length
        start_batch: Starting batch size to test
        max_batch: Maximum batch size to test

    Returns:
        optimal_batch: Largest batch size that fits
    """

    print("=" * 60)
    print("BATCH SIZE OPTIMIZER")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Max length: {max_length}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=5  # Dummy for testing
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Enable FP16 (same as training)
    model = model.half()

    optimal_batch = start_batch

    for batch_size in range(start_batch, max_batch + 1, 8):
        try:
            print(f"\n🧪 Testing batch size: {batch_size}...", end=" ", flush=True)

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Create dummy batch
            dummy_input = {
                'input_ids': torch.randint(0, tokenizer.vocab_size, (batch_size, max_length)).cuda(),
                'attention_mask': torch.ones(batch_size, max_length).cuda(),
                'labels': torch.randint(0, 5, (batch_size, max_length)).cuda(),
            }

            # Forward pass
            outputs = model(**dummy_input)
            loss = outputs.loss

            # Backward pass (this uses the most memory!)
            loss.backward()

            # Clear gradients
            model.zero_grad()

            # Check memory usage
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9

            print(f"✅ SUCCESS! VRAM: {mem_used:.2f}GB used, {mem_reserved:.2f}GB reserved")
            optimal_batch = batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM!")
                break
            else:
                raise e

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 60)
    print(f"✅ OPTIMAL BATCH SIZE: {optimal_batch}")
    print("=" * 60)
    print("\nRecommendations:")
    print(f"  - For training: batch_size = {optimal_batch}")
    print(f"  - For safety (95% VRAM): batch_size = {int(optimal_batch * 0.9)}")
    print(f"  - For multi-task (smaller): batch_size = {optimal_batch // 2}")
    print("\nUpdate your config:")
    print(f'  CONFIG["batch_size"] = {optimal_batch}')

    return optimal_batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find optimal batch size")
    parser.add_argument('--model', type=str, default='dmis-lab/biobert-v1.1',
                       help='Model name')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ No CUDA available! This script requires a GPU.")
        exit(1)

    optimal_batch = test_batch_size(
        model_name=args.model,
        max_length=args.max_length
    )

    print(f"\n🎯 Use batch_size={optimal_batch} in your experiments!")
