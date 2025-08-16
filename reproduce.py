#!/usr/bin/env python3
"""
FlashInfer Prefill Wrapper Reproduction Script

This script loads saved attention metadata files and reproduces the FlashInfer
BatchPrefillWithPagedKVCacheWrapper calls with the same parameters as the original execution.
"""

import torch
import flashinfer
import os
import glob
import argparse
from typing import Dict, Any, List

def load_metadata_file(filepath: str) -> Dict[str, Any]:
    """Load a metadata file and return its contents."""
    try:
        data = torch.load(filepath, map_location='cuda')
        print(f"Loaded metadata from: {filepath}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_mock_kv_cache(num_chunks: int, chunk_size: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
    """Create a mock KV cache for testing purposes."""
    # Create a merged KV cache layout similar to M_LN2HCD
    # Shape: [layers, chunks, 2, heads, chunk_size, head_dim]
    num_layers = 1  # Just one layer for testing
    kv_cache = torch.randn(
        (num_layers, num_chunks, 2, num_kv_heads, chunk_size, head_dim),
        dtype=dtype,
        device='cuda'
    )
    return kv_cache

def create_mock_query(batch_size: int, num_qo_heads: int, head_dim: int, seq_len: int, dtype: torch.dtype = torch.float16):
    """Create a mock query tensor."""
    # Shape: [total_tokens, num_heads, head_dim]
    total_tokens = seq_len * batch_size  # Simplified assumption
    query = torch.randn(total_tokens, num_qo_heads, head_dim, dtype=dtype, device='cuda')
    return query

def reproduce_graph_mode_prefill(metadata_files: List[str], args: argparse.Namespace):
    """Reproduce graph mode with real CUDA graph capture and replay.
    
    Args:
        metadata_files: List of metadata file paths. First file is used for capture, rest for replay.
        args: Command line arguments
    """
    print("\n" + "="*60)
    print("REPRODUCING GRAPH MODE WITH REAL CUDA GRAPHS")
    print("="*60)
    
    if len(metadata_files) == 0:
        print("No metadata files provided for graph mode")
        return
        
    # Use first file for capture
    capture_metadata_file = metadata_files[0]
    replay_metadata_files = metadata_files[1:] if len(metadata_files) > 1 else []
    
    print(f"Capture file: {os.path.basename(capture_metadata_file)}")
    if replay_metadata_files:
        print(f"Replay files: {[os.path.basename(f) for f in replay_metadata_files]}")
    else:
        print("No replay files provided")
    
    # Load capture metadata
    capture_metadata = load_metadata_file(capture_metadata_file)
    if capture_metadata is None:
        print("Failed to load capture metadata")
        return

    # === STEP 1: CUDA GRAPH CAPTURE WITH STATIC ALLOCATIONS ===
    print(f"\n{'='*50}")
    print("STEP 1: REAL CUDA GRAPH CAPTURE (STATIC TENSORS)")
    print(f"{'='*50}")

    # Extract capture metadata and params
    pf_target = capture_metadata.get('pf_target', 0)
    dc_target = capture_metadata.get('dc_target', 0)
    if pf_target == 0:
        print("No prefill requests to reproduce")
        return

    saved_prefill_qo_indptr = capture_metadata['prefill_qo_indptr']
    saved_prefill_paged_kv_indptr = capture_metadata['prefill_paged_kv_indptr']
    saved_prefill_paged_kv_indices = capture_metadata['prefill_paged_kv_indices']
    saved_prefill_paged_kv_last_page_len = capture_metadata['prefill_paged_kv_last_page_len']
    saved_prefill_block_table = capture_metadata['prefill_block_table']

    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    page_size = args.page_size
    params_dtype = torch.float16

    print("Allocating static metadata buffers for graph mode...")
    device = torch.cuda.current_device()

    max_requests = args.max_requests
    max_blocks = args.max_blocks
    max_kv_chunks = args.max_kv_chunks

    static_prefill_qo_indptr_buf = torch.zeros(max_requests + 1, dtype=torch.int32, device=device)
    static_prefill_paged_kv_indptr_buf = torch.zeros(max_requests + 1, dtype=torch.int32, device=device)
    static_prefill_paged_kv_indices_buf = torch.zeros(max_blocks, dtype=torch.int32, device=device)
    static_prefill_paged_kv_last_page_len_buf = torch.zeros(max_requests, dtype=torch.int32, device=device)
    static_prefill_block_table_buf = torch.zeros((max_requests, max_kv_chunks), dtype=torch.int32, device=device)

    # Copy capture metadata into static buffers
    static_prefill_qo_indptr_buf[: len(saved_prefill_qo_indptr)] = saved_prefill_qo_indptr
    static_prefill_paged_kv_indptr_buf[: len(saved_prefill_paged_kv_indptr)] = saved_prefill_paged_kv_indptr
    static_prefill_paged_kv_indices_buf[: len(saved_prefill_paged_kv_indices)] = saved_prefill_paged_kv_indices
    static_prefill_paged_kv_last_page_len_buf[: len(saved_prefill_paged_kv_last_page_len)] = saved_prefill_paged_kv_last_page_len
    if saved_prefill_block_table.numel() > 0:
        rows, cols = saved_prefill_block_table.shape
        static_prefill_block_table_buf[:rows, :cols] = saved_prefill_block_table

    # Workspace and wrapper using static metadata buffers (sliced to capture shapes)
    workspace_buffer = torch.empty(32 * 1024 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    print("Creating BatchPrefillWithPagedKVCacheWrapper (static buffers)...")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        "HND",
        use_cuda_graph=True,
        qo_indptr_buf=static_prefill_qo_indptr_buf[: pf_target + 1],
        paged_kv_indptr_buf=static_prefill_paged_kv_indptr_buf[: pf_target + 1],
        paged_kv_indices_buf=static_prefill_paged_kv_indices_buf,
        paged_kv_last_page_len_buf=static_prefill_paged_kv_last_page_len_buf[: pf_target],
        backend="trtllm-gen",
    )

    # Plan using static metadata buffers
    print("Planning wrapper with static metadata buffers...")
    try:
        wrapper.plan(
            qo_indptr=static_prefill_qo_indptr_buf[: pf_target + 1],
            paged_kv_indptr=static_prefill_paged_kv_indptr_buf[: pf_target + 1],
            paged_kv_indices=static_prefill_paged_kv_indices_buf,
            paged_kv_last_page_len=static_prefill_paged_kv_last_page_len_buf[: pf_target],
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            q_data_type=params_dtype,
            kv_data_type=params_dtype,
            causal=True,
            block_tables=static_prefill_block_table_buf[: pf_target] if saved_prefill_block_table.numel() > 0 else None,
        )
        print("✓ Wrapper planned successfully with static metadata buffers!")
    except Exception as e:
        print(f"✗ Wrapper planning failed: {e}")
        return

    # Allocate static input and output tensors
    total_query_length = int(saved_prefill_qo_indptr[-1].item()) if saved_prefill_qo_indptr.numel() > 0 else pf_target * 10
    static_query = torch.empty(total_query_length, num_qo_heads, head_dim, dtype=params_dtype, device='cuda')

    max_chunks_for_cache = int(saved_prefill_paged_kv_indices.max().item()) + 1 if saved_prefill_paged_kv_indices.numel() > 0 else max_kv_chunks
    static_kv_layer = torch.empty(max_chunks_for_cache, 2, num_kv_heads, page_size, head_dim, dtype=params_dtype, device='cuda')

    static_output = torch.empty_like(static_query)

    # Initialize inputs for capture and copy into static inputs
    init_query = create_mock_query(pf_target, num_qo_heads, head_dim, max(1, total_query_length // pf_target), params_dtype)
    init_kv_cache = create_mock_kv_cache(max_chunks_for_cache, page_size, num_kv_heads, head_dim, params_dtype)
    static_query.copy_(init_query)
    static_kv_layer.copy_(init_kv_cache[0])

    # Create CUDA graph and stream for capture
    capture_stream = torch.cuda.Stream()
    capture_graph = torch.cuda.CUDAGraph()

    # Warmup runs before capture using static inputs and output
    print("Performing warmup runs with static tensors...")
    with torch.cuda.stream(capture_stream):
        warmup_out = wrapper.run(static_query, static_kv_layer)
        static_output.copy_(warmup_out)
        torch.cuda.synchronize()

    # Begin graph capture: run + copy into static_output are captured
    print("Beginning CUDA graph capture...")
    with torch.cuda.stream(capture_stream):
        capture_graph.capture_begin()
        try:
            out = wrapper.run(static_query, static_kv_layer)
            static_output.copy_(out)
            capture_graph.capture_end()
            torch.cuda.synchronize()

        except Exception as e:
            print(f"✗ CUDA graph capture failed: {e}")
            return
        print("✓ CUDA graph capture completed successfully!")

        # Print capture results (from static_output)
        print(f"\nCapture results:")
        print(f"  captured_output shape: {static_output.shape}")
        print(f"  captured_output dtype: {static_output.dtype}")
        print(f"Captured output tensor (first 32 rows):")
        for i in range(min(32, static_output.shape[0])):
            print(f"[{i}]: {static_output[i].flatten()[:10].tolist()}")

    # === STEP 2: CUDA GRAPH REPLAY ===
    if not replay_metadata_files:
        print("\nNo replay files provided, skipping replay phase")
        return

    print(f"\n{'='*50}")
    print("STEP 2: REAL CUDA GRAPH REPLAY (STATIC TENSORS)")
    print(f"{'='*50}")

    # Process replay files using the captured graph
    for i, replay_file in enumerate(replay_metadata_files):
        print(f"\n{'-'*40}")
        print(f"REPLAY {i+1}/{len(replay_metadata_files)}: {os.path.basename(replay_file)}")
        print(f"{'-'*40}")

        replay_metadata = load_metadata_file(replay_file)
        if replay_metadata is None:
            print(f"Failed to load replay metadata: {replay_file}")
            continue

        # Validate shapes are compatible with capture (same pf_target)
        replay_pf_target = replay_metadata.get('pf_target', pf_target)
        if replay_pf_target != pf_target:
            print(f"✗ Replay pf_target ({replay_pf_target}) != capture pf_target ({pf_target}), skipping file")
            continue

        # Copy replay metadata into static buffers
        try:
            r_qo = replay_metadata['prefill_qo_indptr']
            r_kv_indptr = replay_metadata['prefill_paged_kv_indptr']
            r_kv_indices = replay_metadata['prefill_paged_kv_indices']
            r_last_page = replay_metadata['prefill_paged_kv_last_page_len']
            r_block_table = replay_metadata['prefill_block_table']

            print(f"r_qo: {r_qo}")
            print(f"r_kv_indptr: {r_kv_indptr}")
            print(f"r_kv_indices: {r_kv_indices}")
            print(f"r_last_page: {r_last_page}")
            print(f"r_block_table: {r_block_table}")

            static_prefill_qo_indptr_buf[: len(r_qo)] = r_qo
            static_prefill_paged_kv_indptr_buf[: len(r_kv_indptr)] = r_kv_indptr
            static_prefill_paged_kv_indices_buf[: len(r_kv_indices)] = r_kv_indices
            static_prefill_paged_kv_last_page_len_buf[: len(r_last_page)] = r_last_page

            if r_block_table.numel() > 0:
                rows, cols = r_block_table.shape
                static_prefill_block_table_buf[:rows, :cols] = r_block_table
        except Exception as e:
            print(f"✗ Failed to copy replay metadata into static buffers: {e}")
            continue

        # Update static inputs with new random values (or based on replay metadata if desired)
        _ = static_query.normal_(0, 1)
        _ = static_kv_layer.normal_(0, 1)
        static_output[:] = -1

        print(f"static_query")
        for i in range(min(32, static_query.shape[0])):
            print(f"[{i}]: {static_query[i].flatten()[:10].tolist()}")

        # Execute captured graph
        print("Executing captured CUDA graph...")
        try:
            with torch.cuda.stream(capture_stream):
                capture_graph.replay()
                torch.cuda.synchronize()

            print("✓ CUDA graph replay completed successfully!")

            # Print replay results (from static_output)
            print(f"\nReplay results:")
            print(f"  output shape: {static_output.shape}")
            print(f"  output dtype: {static_output.dtype}")
            print(f"Replay output tensor (first 32 rows):")
            for j in range(min(32, static_output.shape[0])):
                print(f"[{j}]: {static_output[j].flatten()[:10].tolist()}")
        except Exception as e:
            print(f"✗ CUDA graph replay failed: {e}")
            continue


        

def reproduce_nongraph_mode_prefill(metadata: Dict[str, Any], args: argparse.Namespace):
    """Reproduce the non-graph mode prefill wrapper call."""
    print("\n" + "="*60)
    print("REPRODUCING NON-GRAPH MODE PREFILL")
    print("="*60)
    
    # Extract metadata using same format as graph mode
    pf_target = metadata.get('pf_target', 0)
    dc_target = metadata.get('dc_target', 0)
    
    print(f"Prefill requests (pf_target): {pf_target}")
    print(f"Decode requests (dc_target): {dc_target}")
    
    if pf_target == 0:
        print("No prefill requests to reproduce")
        return
    
    # Check if prefill tensors exist
    if 'prefill_qo_indptr' not in metadata:
        print("No prefill tensors found in metadata")
        return
    
    # Extract tensors
    prefill_qo_indptr = metadata['prefill_qo_indptr']
    prefill_paged_kv_indptr = metadata['prefill_paged_kv_indptr']
    prefill_paged_kv_indices = metadata['prefill_paged_kv_indices']
    prefill_paged_kv_last_page_len = metadata['prefill_paged_kv_last_page_len']
    prefill_block_table = metadata['prefill_block_table']
    
    print(f"Tensor shapes:")
    print(f"  prefill_qo_indptr: {prefill_qo_indptr.shape}")
    print(f"  prefill_paged_kv_indptr: {prefill_paged_kv_indptr.shape}")
    print(f"  prefill_paged_kv_indices: {prefill_paged_kv_indices.shape}")
    print(f"  prefill_paged_kv_last_page_len: {prefill_paged_kv_last_page_len.shape}")
    print(f"  prefill_block_table: {prefill_block_table.shape}")
    
    # Print detailed metadata values
    print(f"\nDetailed metadata values:")
    print(f"prefill_qo_indptr: {prefill_qo_indptr}")
    print(f"prefill_paged_kv_indptr: {prefill_paged_kv_indptr}")
    print(f"prefill_paged_kv_indices: {prefill_paged_kv_indices}")
    print(f"prefill_paged_kv_last_page_len: {prefill_paged_kv_last_page_len}")
    print(f"prefill_block_table: {prefill_block_table}")
    
    # Print additional metadata if available
    if 'kv_seq_lengths' in metadata:
        print(f"kv_seq_lengths: {metadata['kv_seq_lengths']}")
    if 'full_block_table' in metadata:
        print(f"full_block_table: {metadata['full_block_table']}")
    if 'device_decode_prefill_buf' in metadata:
        print(f"device_decode_prefill_buf: {metadata['device_decode_prefill_buf']}")
    
    # Create workspace buffer
    workspace_buffer = torch.empty(32 * 1024 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    
    # Create wrapper with non-graph mode settings
    print("\nCreating BatchPrefillWithPagedKVCacheWrapper...")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        "HND",  # layout
        use_cuda_graph=False,
        backend="fa2"  # Use same backend as graph mode for consistency
    )
    
    # Mock model parameters
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    page_size = args.page_size
    params_dtype = torch.float16
    
    print(f"\nModel parameters:")
    print(f"  num_qo_heads: {num_qo_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  page_size: {page_size}")
    print(f"  dtype: {params_dtype}")
    
    # Plan the wrapper
    print("\nCalling wrapper.plan()...")
    try:
        wrapper.plan(
            qo_indptr=prefill_qo_indptr,
            paged_kv_indptr=prefill_paged_kv_indptr,
            paged_kv_indices=prefill_paged_kv_indices,
            paged_kv_last_page_len=prefill_paged_kv_last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            q_data_type=params_dtype,
            kv_data_type=params_dtype,
            causal=True,
            block_tables=prefill_block_table,
        )
        print("✓ Plan completed successfully!")
    except Exception as e:
        print(f"✗ Plan failed: {e}")
        return
    
    if not args.skip_run:
        # Create mock inputs for run using same logic as graph mode
        total_query_length = int(prefill_qo_indptr[-1].item()) if prefill_qo_indptr.numel() > 0 else pf_target * 10
        query = create_mock_query(pf_target, num_qo_heads, head_dim, max(1, total_query_length // pf_target), params_dtype)
        
        # Create mock KV cache
        max_chunks = int(prefill_paged_kv_indices.max().item()) + 1 if prefill_paged_kv_indices.numel() > 0 else 100
        kv_cache = create_mock_kv_cache(max_chunks, page_size, num_kv_heads, head_dim, params_dtype)
        
        print(f"\nCreated mock inputs:")
        print(f"  query shape: {query.shape}")
        print(f"  kv_cache shape: {kv_cache.shape}")
        
        # Print input query details
        print(f"\nInput query details:")
        print(f"  query dtype: {query.dtype}")
        print(f"  query device: {query.device}")
        print(f"  query shape: {query.shape}")
        print(f"Input query tensor (first 32 rows):")
        for i in range(min(32, query.shape[0])):
            print(f"[{i}]: {query[i].flatten()[:10].tolist()}")
        
        # Run the wrapper
        print("\nCalling wrapper.run()...")
        try:
            output = wrapper.run(query, kv_cache[0])  # layer 0
            print(f"✓ Run completed successfully!")
            print(f"  output shape: {output.shape}")
            
            # Print output details
            print(f"\nOutput details:")
            print(f"  output dtype: {output.dtype}")
            print(f"  output device: {output.device}")
            print(f"  output shape: {output.shape}")
            print(f"Output tensor (first 32 rows):")
            for i in range(min(32, output.shape[0])):
                print(f"[{i}]: {output[i].flatten()[:10].tolist()}")
            
        except Exception as e:
            print(f"✗ Run failed: {e}")

def find_metadata_files(directory: str) -> List[str]:
    """Find all metadata files in the directory."""
    pattern = os.path.join(directory, "*_metadata_*.pt")
    files = glob.glob(pattern)
    files.sort()  # Sort by filename for consistent order
    return files

def main():
    parser = argparse.ArgumentParser(description="Reproduce FlashInfer prefill wrapper calls from saved metadata")
    parser.add_argument("--metadata-dir", type=str, default="attn_metadata_saves", 
                       help="Directory containing saved metadata files")
    parser.add_argument("--file", type=str, help="Specific metadata file to process")
    parser.add_argument("--mode", type=str, choices=['graph', 'non_graph'], 
                       help="Force specify mode: 'graph' for CUDA graph capture/replay, 'non_graph' for direct execution")
    parser.add_argument("--skip-run", action="store_true", 
                       help="Skip the wrapper.run() call, only test plan()")
    parser.add_argument("--num-qo-heads", type=int, default=32, 
                       help="Number of query/output heads")
    parser.add_argument("--num-kv-heads", type=int, default=8, 
                       help="Number of key/value heads")
    parser.add_argument("--head-dim", type=int, default=128, 
                       help="Head dimension")
    parser.add_argument("--page-size", type=int, default=16, 
                       help="Page size (chunk size tokens)")
    parser.add_argument("--max-files", type=int, default=20,
                       help="Maximum number of files to process")
    
    # Graph mode static allocation parameters
    parser.add_argument("--max-requests", type=int, default=128,
                       help="Maximum requests for static allocation in graph mode")
    parser.add_argument("--max-blocks", type=int, default=65536,
                       help="Maximum blocks for static allocation in graph mode")
    parser.add_argument("--max-kv-chunks", type=int, default=32,
                       help="Maximum KV chunks for static allocation in graph mode")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires CUDA.")
        return
    
    print("FlashInfer Prefill Wrapper Reproduction Script")
    print("=" * 60)
    
    if args.file:
        # Process specific file
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        files = [args.file]
    else:
        # Find all metadata files
        if not os.path.exists(args.metadata_dir):
            print(f"Metadata directory not found: {args.metadata_dir}")
            return
        
        files = find_metadata_files(args.metadata_dir)
        if not files:
            print(f"No metadata files found in {args.metadata_dir}")
            return
        
        print(f"Found {len(files)} metadata files")
        for f in files:
            print(f)
        files = files[:args.max_files]  # Limit number of files
    
    if not args.mode:
        print("Error: --mode argument is required. Use --mode graph or --mode non_graph")
        return
    
    print(f"Force mode: {args.mode}")
    
    if args.mode == 'graph':
        # Process files in graph mode (first file for capture, rest for replay)
        if len(files) == 0:
            print("No metadata files provided for graph mode")
            return
        
        print(f"\n{'='*80}")
        print(f"PROCESSING FILES IN GRAPH MODE ({len(files)} files)")
        print(f"{'='*80}")
        reproduce_graph_mode_prefill(files, args)
        
    elif args.mode == 'non_graph':
        # Process files individually in non-graph mode
        for i, filepath in enumerate(files):
            print(f"\n{'='*80}")
            print(f"PROCESSING FILE {i+1}/{len(files)} IN NON-GRAPH MODE: {os.path.basename(filepath)}")
            print(f"{'='*80}")
            
            metadata = load_metadata_file(filepath)
            if metadata is None:
                continue
            
            reproduce_nongraph_mode_prefill(metadata, args)
            print(f"\nCompleted processing {os.path.basename(filepath)}")

if __name__ == "__main__":
    main()

