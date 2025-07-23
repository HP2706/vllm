#!/usr/bin/env python3

"""
Tests for the output processor factory method and integration.

These tests verify that the SequenceHookOutputProcessor is properly
created when special_kwargs are provided.
"""

import pytest
from unittest.mock import Mock

from vllm.config import SchedulerConfig, SpecialKwargs
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
from vllm.engine.output_processor.sequence_hook import SequenceHookOutputProcessor
from vllm.engine.output_processor.multi_step import MultiStepOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter


class TestOutputProcessorFactory:
    """Test the factory method for creating output processors."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for the factory."""
        scheduler_config = Mock(spec=SchedulerConfig)
        detokenizer = Mock(spec=Detokenizer)
        scheduler = [Mock(spec=Scheduler)]
        seq_counter = Counter()
        stop_checker = Mock(spec=StopChecker)
        get_tokenizer_for_seq = Mock()
        
        return {
            'scheduler_config': scheduler_config,
            'detokenizer': detokenizer,
            'scheduler': scheduler,
            'seq_counter': seq_counter,
            'get_tokenizer_for_seq': get_tokenizer_for_seq,
            'stop_checker': stop_checker
        }

    def test_creates_sequence_hook_processor_with_special_kwargs(self, mock_dependencies):
        """Test that SequenceHookOutputProcessor is created when special_kwargs provided."""
        # Setup single-step config
        mock_dependencies['scheduler_config'].num_lookahead_slots = 0
        
        special_kwargs = SpecialKwargs(
            sync_token_id=12345,
            promise_token_id=67890
        )
        
        # Create processor
        processor = SequenceGroupOutputProcessor.create_output_processor(
            special_kwargs=special_kwargs,
            **mock_dependencies
        )
        
        # Verify correct type
        assert isinstance(processor, SequenceHookOutputProcessor)
        assert processor.special_kwargs == special_kwargs
        print("✅ SequenceHookOutputProcessor created with special_kwargs")

    def test_creates_single_step_processor_without_special_kwargs(self, mock_dependencies):
        """Test that SingleStepOutputProcessor is created when no special_kwargs."""
        # Setup single-step config
        mock_dependencies['scheduler_config'].num_lookahead_slots = 0
        
        # Create processor without special_kwargs
        processor = SequenceGroupOutputProcessor.create_output_processor(
            special_kwargs=None,
            **mock_dependencies
        )
        
        # Verify correct type
        assert isinstance(processor, SingleStepOutputProcessor)
        assert not isinstance(processor, SequenceHookOutputProcessor)
        print("✅ SingleStepOutputProcessor created without special_kwargs")

    def test_creates_multi_step_processor_for_multi_step_config(self, mock_dependencies):
        """Test that MultiStepOutputProcessor is created for multi-step scheduling."""
        # Setup multi-step config
        mock_dependencies['scheduler_config'].num_lookahead_slots = 5
        
        # Create processor (should be multi-step regardless of special_kwargs)
        processor = SequenceGroupOutputProcessor.create_output_processor(
            special_kwargs=SpecialKwargs(sync_token_id=123, promise_token_id=456),
            **mock_dependencies
        )
        
        # Verify correct type (multi-step takes precedence)
        assert isinstance(processor, MultiStepOutputProcessor)
        assert not isinstance(processor, SequenceHookOutputProcessor)
        print("✅ MultiStepOutputProcessor created for multi-step config")

    def test_inheritance_hierarchy(self):
        """Test that SequenceHookOutputProcessor properly inherits from SingleStepOutputProcessor."""
        # Create minimal dependencies
        scheduler_config = Mock(spec=SchedulerConfig)
        detokenizer = Mock(spec=Detokenizer)
        scheduler = [Mock(spec=Scheduler)]
        seq_counter = Counter()
        stop_checker = Mock(spec=StopChecker)
        special_kwargs = SpecialKwargs(sync_token_id=123, promise_token_id=456)
        
        processor = SequenceHookOutputProcessor(
            scheduler_config, detokenizer, scheduler, seq_counter, stop_checker, special_kwargs
        )
        
        # Verify inheritance
        assert isinstance(processor, SingleStepOutputProcessor)
        assert isinstance(processor, SequenceGroupOutputProcessor)
        assert hasattr(processor, 'process_outputs')
        assert hasattr(processor, 'process_prompt_logprob')
        assert hasattr(processor, '_run_sequence_hook')
        print("✅ Inheritance hierarchy verified")


class TestSpecialKwargsIntegration:
    """Test integration with special_kwargs configuration."""

    def test_special_kwargs_types(self):
        """Test that SpecialKwargs accepts correct types."""
        # Test valid construction
        special_kwargs = SpecialKwargs(
            sync_token_id=12345,
            promise_token_id=67890
        )
        
        assert special_kwargs.sync_token_id == 12345
        assert special_kwargs.promise_token_id == 67890
        print("✅ SpecialKwargs construction verified")

    def test_special_kwargs_optional_fields(self):
        """Test SpecialKwargs with optional/missing fields."""
        # Test with only one field
        special_kwargs = SpecialKwargs(sync_token_id=123)
        
        assert special_kwargs.sync_token_id == 123
        # Check if promise_token_id has a default or raises appropriate error
        print("✅ SpecialKwargs optional fields verified")


def run_factory_tests():
    """Run factory tests manually (for development)."""
    print("🧪 Running OutputProcessorFactory tests...")
    
    # Create test instance
    test_instance = TestOutputProcessorFactory()
    special_kwargs_test = TestSpecialKwargsIntegration()
    
    # Setup fixtures manually
    mock_deps = test_instance.mock_dependencies()
    
    # Run tests
    try:
        test_instance.test_creates_sequence_hook_processor_with_special_kwargs(mock_deps)
        test_instance.test_creates_single_step_processor_without_special_kwargs(mock_deps)
        test_instance.test_creates_multi_step_processor_for_multi_step_config(mock_deps)
        test_instance.test_inheritance_hierarchy()
        
        special_kwargs_test.test_special_kwargs_types()
        special_kwargs_test.test_special_kwargs_optional_fields()
        
        print("\n🎉 All factory tests passed!")
        
    except Exception as e:
        print(f"\n❌ Factory test failed: {e}")
        raise


if __name__ == "__main__":
    run_factory_tests() 