#!/usr/bin/env python3
"""
Test script for the special token detection and hook mechanism.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'vllm'))

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import CompletionSequenceGroupOutput, SequenceOutput, SequenceGroupMetadata
from vllm.config import SpecialKwargs


def test_special_token_detection():
    """Test the special token detection functionality."""
    
    # Define special token IDs
    sync_token_id = 12345
    promise_token_id = 67890
    normal_token_id = 100
    
    # Create mock sequence outputs
    normal_output = SequenceOutput(parent_seq_id=0, output_token=normal_token_id, logprobs={})
    sync_output = SequenceOutput(parent_seq_id=1, output_token=sync_token_id, logprobs={})
    promise_output = SequenceOutput(parent_seq_id=2, output_token=promise_token_id, logprobs={})
    
    # Create completion sequence group outputs
    normal_group = CompletionSequenceGroupOutput(samples=[normal_output], prompt_logprobs=None)
    sync_group = CompletionSequenceGroupOutput(samples=[sync_output], prompt_logprobs=None)
    promise_group = CompletionSequenceGroupOutput(samples=[promise_output], prompt_logprobs=None)
    
    # Create sampler outputs
    normal_sampler_output = SamplerOutput(outputs=[normal_group])
    sync_sampler_output = SamplerOutput(outputs=[sync_group])
    promise_sampler_output = SamplerOutput(outputs=[promise_group])
    mixed_sampler_output = SamplerOutput(outputs=[normal_group, sync_group, promise_group])
    
    # Test 1: Normal token should not trigger special token detection
    assert not normal_sampler_output.has_sync_token(sync_token_id)
    assert not normal_sampler_output.has_promise_token(promise_token_id)
    print("✓ Test 1 passed: Normal tokens don't trigger special detection")
    
    # Test 2: Sync token should be detected
    assert sync_sampler_output.has_sync_token(sync_token_id)
    assert not sync_sampler_output.has_promise_token(promise_token_id)
    sync_sequences = sync_sampler_output.get_sync_sequences(sync_token_id)
    assert len(sync_sequences) == 1
    assert sync_sequences[0] == (0, 0, 1)  # (seq_group_idx, sample_idx, parent_seq_id)
    print("✓ Test 2 passed: Sync token detection works")
    
    # Test 3: Promise token should be detected
    assert promise_sampler_output.has_promise_token(promise_token_id)
    assert not promise_sampler_output.has_sync_token(sync_token_id)
    promise_sequences = promise_sampler_output.get_promise_sequences(promise_token_id)
    assert len(promise_sequences) == 1
    assert promise_sequences[0] == (0, 0, 2)  # (seq_group_idx, sample_idx, parent_seq_id)
    print("✓ Test 3 passed: Promise token detection works")
    
    # Test 4: Mixed output should detect both
    assert mixed_sampler_output.has_sync_token(sync_token_id)
    assert mixed_sampler_output.has_promise_token(promise_token_id)
    mixed_sync_sequences = mixed_sampler_output.get_sync_sequences(sync_token_id)
    mixed_promise_sequences = mixed_sampler_output.get_promise_sequences(promise_token_id)
    assert len(mixed_sync_sequences) == 1
    assert len(mixed_promise_sequences) == 1
    assert mixed_sync_sequences[0] == (1, 0, 1)  # sync token is in the second group
    assert mixed_promise_sequences[0] == (2, 0, 2)  # promise token is in the third group
    print("✓ Test 4 passed: Mixed output detection works")
    
    print("\n🎉 All tests passed! Special token detection is working correctly.")


def test_special_kwargs():
    """Test the SpecialKwargs functionality."""
    
    # Create special kwargs
    special_kwargs = SpecialKwargs()
    special_kwargs.sync_token_id = 12345
    special_kwargs.promise_token_id = 67890
    
    assert special_kwargs.sync_token_id == 12345
    assert special_kwargs.promise_token_id == 67890
    print("✓ SpecialKwargs works correctly")


if __name__ == "__main__":
    print("Testing special token detection mechanism...\n")
    test_special_token_detection()
    print()
    test_special_kwargs()
    print("\n✅ All tests completed successfully!") 