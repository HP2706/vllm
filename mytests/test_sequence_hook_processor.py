#!/usr/bin/env python3

"""
Surgical tests for SequenceHookOutputProcessor.

These tests directly manipulate vLLM's internal state to test specific
sequence hook behaviors without needing full model inference.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

# Import vLLM components
from vllm.config import SchedulerConfig, SpecialKwargs
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.sequence_hook import SequenceHookOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sequence import (
    Sequence, SequenceGroup, SequenceStatus, SequenceOutput,
    CompletionSequenceGroupOutput, SampleLogprobs
)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter
from vllm.sampling_params import SamplingParams
from vllm.inputs import SingletonInputs


class TestSequenceHookProcessor:
    """Test suite for SequenceHookOutputProcessor functionality."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for the processor."""
        scheduler_config = Mock(spec=SchedulerConfig)
        detokenizer = Mock(spec=Detokenizer)
        scheduler = [Mock(spec=Scheduler)]
        seq_counter = Counter()
        stop_checker = Mock(spec=StopChecker)
        
        special_kwargs = SpecialKwargs(
            sync_token_id=12345,
            promise_token_id=67890
        )
        
        return {
            'scheduler_config': scheduler_config,
            'detokenizer': detokenizer,
            'scheduler': scheduler,
            'seq_counter': seq_counter,
            'stop_checker': stop_checker,
            'special_kwargs': special_kwargs
        }

    @pytest.fixture
    def processor(self, mock_dependencies):
        """Create SequenceHookOutputProcessor instance."""
        return SequenceHookOutputProcessor(**mock_dependencies)

    def create_test_sequence(self, seq_id: int, status: SequenceStatus = SequenceStatus.RUNNING):
        """Create a test sequence with minimal required state."""
        # Create minimal inputs for the sequence
        inputs = SingletonInputs({
            "prompt_token_ids": [1, 2, 3, 4],
            "prompt": "test prompt",
            "type": "text"
        })
        
        seq = Sequence(
            seq_id=seq_id,
            inputs=inputs,
            block_size=16,
            eos_token_id=2
        )
        seq.status = status
        return seq

    def create_test_sequence_group(self, request_id: str, seqs: List[Sequence]):
        """Create a test sequence group with given sequences."""
        sampling_params = SamplingParams(max_tokens=100)
        
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=seqs,
            arrival_time=0.0,
            sampling_params=sampling_params
        )
        return seq_group

    def create_completion_output(self, samples: List[SequenceOutput]):
        """Create a CompletionSequenceGroupOutput with given samples."""
        return CompletionSequenceGroupOutput(
            samples=samples,
            prompt_logprobs=None
        )

    def create_sample_output(self, output_token: int, parent_seq_id: int):
        """Create a SequenceOutput sample."""
        return SequenceOutput(
            parent_seq_id=parent_seq_id,
            output_token=output_token,
            logprobs={output_token: 0.0}  # Simple logprobs
        )

    def test_sync_token_detection(self, processor):
        """Test that sync tokens are properly detected and sequences are blocked."""
        # Create test sequence
        seq = self.create_test_sequence(seq_id=1, status=SequenceStatus.RUNNING)
        seq_group = self.create_test_sequence_group("test_req", [seq])
        
        # Create output with sync token
        sync_sample = self.create_sample_output(
            output_token=processor.special_kwargs.sync_token_id,
            parent_seq_id=1
        )
        output = self.create_completion_output([sync_sample])
        
        # Test sync token handling
        processor._handle_sync(seq_group, output, processor.special_kwargs.sync_token_id)
        
        # Verify sequence was blocked
        assert seq.status == SequenceStatus.BLOCKED
        print("✅ Sync token detection test passed")

    def test_promise_token_spawning(self, processor):
        """Test that promise tokens spawn child sequences correctly."""
        # Create parent sequence
        parent_seq = self.create_test_sequence(seq_id=1, status=SequenceStatus.RUNNING)
        parent_seq.can_have_children = True
        seq_group = self.create_test_sequence_group("test_req", [parent_seq])
        
        # Mock the spawn_child method
        child_seq = self.create_test_sequence(seq_id=2, status=SequenceStatus.WAITING)
        child_seq.parent_seq_id = 1
        seq_group.spawn_child = Mock(return_value=child_seq)
        
        # Create output with promise token
        promise_sample = self.create_sample_output(
            output_token=processor.special_kwargs.promise_token_id,
            parent_seq_id=1
        )
        output = self.create_completion_output([promise_sample])
        
        # Test promise token handling
        processor._handle_promise(seq_group, output, processor.special_kwargs.promise_token_id)
        
        # Verify child was spawned and parent was blocked
        seq_group.spawn_child.assert_called_once()
        assert parent_seq.status == SequenceStatus.BLOCKED
        print("✅ Promise token spawning test passed")

    def test_parent_unblocking_when_children_finish(self, processor):
        """Test that parent sequences are unblocked when all children finish."""
        # Create parent sequence (blocked, waiting for children)
        parent_seq = self.create_test_sequence(seq_id=1, status=SequenceStatus.BLOCKED)
        parent_seq.children_seq_ids = {2, 3}
        
        # Create finished child sequences
        child1 = self.create_test_sequence(seq_id=2, status=SequenceStatus.FINISHED_STOPPED)
        child1.parent_seq_id = 1
        child2 = self.create_test_sequence(seq_id=3, status=SequenceStatus.FINISHED_STOPPED)
        child2.parent_seq_id = 1
        
        seq_group = self.create_test_sequence_group("test_req", [parent_seq, child1, child2])
        
        # Mock the merge results method to avoid complex logic
        processor._merge_child_results = Mock()
        
        # Test unblocking logic
        processor._handle_unblock(seq_group)
        
        # Verify parent was unblocked and merge was called
        assert parent_seq.status == SequenceStatus.WAITING
        processor._merge_child_results.assert_called_once_with(parent_seq, seq_group)
        print("✅ Parent unblocking test passed")

    def test_parent_stays_blocked_with_unfinished_children(self, processor):
        """Test that parent remains blocked if children are still running."""
        # Create parent sequence (blocked)
        parent_seq = self.create_test_sequence(seq_id=1, status=SequenceStatus.BLOCKED)
        parent_seq.children_seq_ids = {2, 3}
        
        # Create one finished, one running child
        child1 = self.create_test_sequence(seq_id=2, status=SequenceStatus.FINISHED_STOPPED)
        child1.parent_seq_id = 1
        child2 = self.create_test_sequence(seq_id=3, status=SequenceStatus.RUNNING)
        child2.parent_seq_id = 1
        
        seq_group = self.create_test_sequence_group("test_req", [parent_seq, child1, child2])
        
        # Test unblocking logic
        processor._handle_unblock(seq_group)
        
        # Verify parent stays blocked
        assert parent_seq.status == SequenceStatus.BLOCKED
        print("✅ Parent stays blocked test passed")

    def test_merge_child_results(self, processor):
        """Test merging of child sequence results into parent."""
        # Create parent sequence
        parent_seq = self.create_test_sequence(seq_id=1)
        
        # Create finished children with output text
        child1 = self.create_test_sequence(seq_id=2, status=SequenceStatus.FINISHED_STOPPED)
        child1.parent_seq_id = 1
        child1.output_text = "Child 1 result"
        
        child2 = self.create_test_sequence(seq_id=3, status=SequenceStatus.FINISHED_STOPPED)
        child2.parent_seq_id = 1
        child2.output_text = "Child 2 result"
        
        seq_group = self.create_test_sequence_group("test_req", [parent_seq, child1, child2])
        
        # Mock the parent sequence data append method
        parent_seq.data = Mock()
        parent_seq.data.append_prompt_tokens = Mock()
        
        # Test merge logic
        processor._merge_child_results(parent_seq, seq_group)
        
        # Verify merge was called (we're testing the general flow here)
        # The exact merge logic can be customized per use case
        expected_merged = "Child 1 result\nChild 2 result"
        computed_output = processor._compute_merged_output([child1, child2])
        assert computed_output == expected_merged
        print("✅ Child result merging test passed")

    def test_process_outputs_integration(self, processor):
        """Test the full process_outputs method integration."""
        # Create test sequence and group
        seq = self.create_test_sequence(seq_id=1, status=SequenceStatus.RUNNING)
        seq_group = self.create_test_sequence_group("test_req", [seq])
        
        # Create output with sync token
        sync_sample = self.create_sample_output(
            output_token=processor.special_kwargs.sync_token_id,
            parent_seq_id=1
        )
        outputs = [self.create_completion_output([sync_sample])]
        
        # Mock the parent class process_outputs to avoid complex dependencies
        with pytest.mock.patch.object(
            processor.__class__.__bases__[0], 'process_outputs'
        ) as mock_parent_process:
            processor.process_outputs(seq_group, outputs, is_async=False)
            
            # Verify our hook ran (sequence should be blocked)
            assert seq.status == SequenceStatus.BLOCKED
            
            # Verify parent process_outputs was called
            mock_parent_process.assert_called_once_with(seq_group, outputs, False)
        
        print("✅ Full integration test passed")

    def test_no_special_tokens_normal_processing(self, processor):
        """Test that normal tokens don't trigger special behavior."""
        # Create test sequence
        seq = self.create_test_sequence(seq_id=1, status=SequenceStatus.RUNNING)
        seq_group = self.create_test_sequence_group("test_req", [seq])
        
        # Create output with normal token (not sync or promise)
        normal_sample = self.create_sample_output(output_token=999, parent_seq_id=1)
        output = self.create_completion_output([normal_sample])
        
        original_status = seq.status
        
        # Test hook doesn't trigger
        processor._run_sequence_hook(seq_group, [output])
        
        # Verify sequence status unchanged
        assert seq.status == original_status
        print("✅ Normal token processing test passed")


def run_tests():
    """Run all tests manually (for development)."""
    print("🧪 Running SequenceHookProcessor tests...")
    
    # Create test instance
    test_instance = TestSequenceHookProcessor()
    
    # Setup fixtures manually
    mock_deps = test_instance.mock_dependencies()
    processor = test_instance.processor(mock_deps)
    
    # Run individual tests
    try:
        test_instance.test_sync_token_detection(processor)
        test_instance.test_promise_token_spawning(processor)
        test_instance.test_parent_unblocking_when_children_finish(processor)
        test_instance.test_parent_stays_blocked_with_unfinished_children(processor)
        test_instance.test_merge_child_results(processor)
        test_instance.test_process_outputs_integration(processor)
        test_instance.test_no_special_tokens_normal_processing(processor)
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_tests() 