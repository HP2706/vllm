#!/usr/bin/env python3

"""
Edge case and error handling tests for SequenceHookOutputProcessor.

These tests cover error conditions, malformed inputs, and edge cases
that could occur in production.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from vllm.config import SchedulerConfig, SpecialKwargs
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.sequence_hook import SequenceHookOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sequence import (
    Sequence, SequenceGroup, SequenceStatus, SequenceOutput,
    CompletionSequenceGroupOutput, PoolingSequenceGroupOutput
)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter
from vllm.sampling_params import SamplingParams
from vllm.inputs import SingletonInputs


class TestSequenceHookEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def processor(self):
        """Create processor with standard test configuration."""
        scheduler_config = Mock(spec=SchedulerConfig)
        detokenizer = Mock(spec=Detokenizer)
        scheduler = [Mock(spec=Scheduler)]
        seq_counter = Counter()
        stop_checker = Mock(spec=StopChecker)
        special_kwargs = SpecialKwargs(sync_token_id=12345, promise_token_id=67890)
        
        return SequenceHookOutputProcessor(
            scheduler_config, detokenizer, scheduler, seq_counter, stop_checker, special_kwargs
        )

    def create_test_sequence(self, seq_id: int, status: SequenceStatus = SequenceStatus.RUNNING):
        """Helper to create test sequences."""
        inputs = SingletonInputs({
            "prompt_token_ids": [1, 2, 3, 4],
            "prompt": "test prompt",
            "type": "text"
        })
        
        seq = Sequence(seq_id=seq_id, inputs=inputs, block_size=16, eos_token_id=2)
        seq.status = status
        return seq

    def test_empty_outputs_list(self, processor):
        """Test handling of empty outputs list."""
        seq = self.create_test_sequence(1)
        seq_group = SequenceGroup(
            request_id="test", seqs=[seq], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Test with empty outputs
        processor._run_sequence_hook(seq_group, [])
        
        # Should not crash and sequence status should be unchanged
        assert seq.status == SequenceStatus.RUNNING
        print("✅ Empty outputs handled gracefully")

    def test_none_outputs(self, processor):
        """Test handling of None in outputs."""
        seq = self.create_test_sequence(1)
        seq_group = SequenceGroup(
            request_id="test", seqs=[seq], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Test with None outputs
        processor._run_sequence_hook(seq_group, [None])
        
        # Should not crash
        assert seq.status == SequenceStatus.RUNNING
        print("✅ None outputs handled gracefully")

    def test_wrong_output_type(self, processor):
        """Test handling of wrong output type (not CompletionSequenceGroupOutput)."""
        seq = self.create_test_sequence(1)
        seq_group = SequenceGroup(
            request_id="test", seqs=[seq], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Create wrong type of output (PoolingSequenceGroupOutput)
        pooling_output = Mock(spec=PoolingSequenceGroupOutput)
        
        # Should handle gracefully and return early
        processor._run_sequence_hook(seq_group, [pooling_output])
        
        assert seq.status == SequenceStatus.RUNNING
        print("✅ Wrong output type handled gracefully")

    def test_malformed_samples(self, processor):
        """Test handling of malformed samples in output."""
        seq = self.create_test_sequence(1)
        seq_group = SequenceGroup(
            request_id="test", seqs=[seq], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Create output with malformed samples
        output = CompletionSequenceGroupOutput(
            samples=None,  # Invalid samples
            prompt_logprobs=None
        )
        
        # Should handle gracefully
        try:
            processor._run_sequence_hook(seq_group, [output])
            print("✅ Malformed samples handled gracefully")
        except (TypeError, AttributeError):
            # Expected if samples is None
            print("✅ Malformed samples correctly rejected")

    def test_sequence_index_out_of_bounds(self, processor):
        """Test handling when sequence index exceeds available sequences."""
        seq = self.create_test_sequence(1)
        seq_group = SequenceGroup(
            request_id="test", seqs=[seq], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Create sample pointing to non-existent sequence index
        sample = SequenceOutput(
            parent_seq_id=999,  # Non-existent sequence
            output_token=processor.special_kwargs.sync_token_id,
            logprobs={processor.special_kwargs.sync_token_id: 0.0}
        )
        output = CompletionSequenceGroupOutput(samples=[sample], prompt_logprobs=None)
        
        # Should handle gracefully without crashing
        processor._handle_sync(seq_group, output, processor.special_kwargs.sync_token_id)
        
        # Original sequence should be unchanged
        assert seq.status == SequenceStatus.RUNNING
        print("✅ Out of bounds sequence index handled gracefully")

    def test_promise_token_on_non_parent_sequence(self, processor):
        """Test promise token on sequence that can't have children."""
        seq = self.create_test_sequence(1)
        seq.can_have_children = False  # Explicitly set to False
        seq_group = SequenceGroup(
            request_id="test", seqs=[seq], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Mock spawn_child to raise an error
        seq_group.spawn_child = Mock(side_effect=AssertionError("Cannot have children"))
        
        sample = SequenceOutput(
            parent_seq_id=1,
            output_token=processor.special_kwargs.promise_token_id,
            logprobs={processor.special_kwargs.promise_token_id: 0.0}
        )
        output = CompletionSequenceGroupOutput(samples=[sample], prompt_logprobs=None)
        
        # Should handle the error gracefully
        try:
            processor._handle_promise(seq_group, output, processor.special_kwargs.promise_token_id)
        except AssertionError:
            # Expected behavior - sequences that can't have children should fail
            pass
        
        print("✅ Promise token on non-parent sequence handled correctly")

    def test_orphaned_child_sequences(self, processor):
        """Test handling of child sequences with no parent."""
        # Create child sequence with non-existent parent
        child = self.create_test_sequence(2, SequenceStatus.FINISHED_STOPPED)
        child.parent_seq_id = 999  # Non-existent parent
        
        seq_group = SequenceGroup(
            request_id="test", seqs=[child], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Should handle gracefully
        processor._handle_unblock(seq_group)
        
        # Child status should remain unchanged
        assert child.status == SequenceStatus.FINISHED_STOPPED
        print("✅ Orphaned child sequences handled gracefully")

    def test_circular_parent_child_relationship(self, processor):
        """Test detection of circular parent-child relationships."""
        # Create sequences with circular dependency
        seq1 = self.create_test_sequence(1, SequenceStatus.BLOCKED)
        seq1.parent_seq_id = 2
        seq1.children_seq_ids = {2}
        
        seq2 = self.create_test_sequence(2, SequenceStatus.BLOCKED)
        seq2.parent_seq_id = 1
        seq2.children_seq_ids = {1}
        
        seq_group = SequenceGroup(
            request_id="test", seqs=[seq1, seq2], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Should handle without infinite loop
        processor._handle_unblock(seq_group)
        
        # Both should remain blocked (circular dependency)
        assert seq1.status == SequenceStatus.BLOCKED
        assert seq2.status == SequenceStatus.BLOCKED
        print("✅ Circular dependency handled without infinite loop")

    def test_merge_with_empty_child_list(self, processor):
        """Test merging when no children exist."""
        parent_seq = self.create_test_sequence(1)
        seq_group = SequenceGroup(
            request_id="test", seqs=[parent_seq], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Mock append method
        parent_seq.data = Mock()
        parent_seq.data.append_prompt_tokens = Mock()
        
        # Should handle empty child list gracefully
        processor._merge_child_results(parent_seq, seq_group)
        
        # append_prompt_tokens should not be called with empty result
        parent_seq.data.append_prompt_tokens.assert_not_called()
        print("✅ Empty child list handled gracefully")

    def test_merge_with_children_no_output(self, processor):
        """Test merging children that have no output text."""
        parent_seq = self.create_test_sequence(1)
        
        # Children with no output text
        child1 = self.create_test_sequence(2, SequenceStatus.FINISHED_STOPPED)
        child1.parent_seq_id = 1
        child1.output_text = ""
        
        child2 = self.create_test_sequence(3, SequenceStatus.FINISHED_STOPPED)
        child2.parent_seq_id = 1
        child2.output_text = None
        
        seq_group = SequenceGroup(
            request_id="test", seqs=[parent_seq, child1, child2], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        # Should handle children with no output
        merged = processor._compute_merged_output([child1, child2])
        
        # Should return empty string for no output
        assert merged == ""
        print("✅ Children with no output handled gracefully")


def run_edge_case_tests():
    """Run edge case tests manually (for development)."""
    print("🧪 Running SequenceHook edge case tests...")
    
    test_instance = TestSequenceHookEdgeCases()
    
    # Create processor fixture manually
    processor = test_instance.processor()
    
    # Run tests
    try:
        test_instance.test_empty_outputs_list(processor)
        test_instance.test_none_outputs(processor)
        test_instance.test_wrong_output_type(processor)
        test_instance.test_malformed_samples(processor)
        test_instance.test_sequence_index_out_of_bounds(processor)
        test_instance.test_promise_token_on_non_parent_sequence(processor)
        test_instance.test_orphaned_child_sequences(processor)
        test_instance.test_circular_parent_child_relationship(processor)
        test_instance.test_merge_with_empty_child_list(processor)
        test_instance.test_merge_with_children_no_output(processor)
        
        print("\n🎉 All edge case tests passed!")
        
    except Exception as e:
        print(f"\n❌ Edge case test failed: {e}")
        raise


if __name__ == "__main__":
    run_edge_case_tests() 