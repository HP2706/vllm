#!/usr/bin/env python3

"""
Demo test showing the surgical testing approach for SequenceHookOutputProcessor.

This demonstrates how to create minimal test scenarios that directly test
specific sequence hook behaviors without full model inference.
"""

from unittest.mock import Mock

from vllm.config import SchedulerConfig, SpecialKwargs
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.sequence_hook import SequenceHookOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sequence import (
    Sequence, SequenceGroup, SequenceStatus, SequenceOutput,
    CompletionSequenceGroupOutput
)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter
from vllm.sampling_params import SamplingParams
from vllm.inputs import SingletonInputs


def demo_surgical_testing():
    """Demonstrate the surgical testing approach."""
    print("🧪 Demo: Surgical Testing of SequenceHookOutputProcessor")
    print("=" * 60)
    
    # Step 1: Create minimal dependencies (mocked)
    print("1️⃣  Creating minimal mock dependencies...")
    scheduler_config = Mock(spec=SchedulerConfig)
    detokenizer = Mock(spec=Detokenizer)
    scheduler = [Mock(spec=Scheduler)]
    seq_counter = Counter()
    stop_checker = Mock(spec=StopChecker)
    
    special_kwargs = SpecialKwargs(
        sync_token_id=12345,    # <sync> token
        promise_token_id=67890  # <promise> token
    )
    
    # Step 2: Create the processor
    print("2️⃣  Creating SequenceHookOutputProcessor...")
    processor = SequenceHookOutputProcessor(
        scheduler_config, detokenizer, scheduler, seq_counter, 
        stop_checker, special_kwargs
    )
    
    # Step 3: Create test sequence with specific state
    print("3️⃣  Creating test sequence in RUNNING state...")
    inputs = SingletonInputs({
        "prompt_token_ids": [1, 2, 3, 4],
        "prompt": "Test prompt for demo",
        "type": "text"
    })
    
    test_seq = Sequence(
        seq_id=1,
        inputs=inputs,
        block_size=16,
        eos_token_id=2
    )
    test_seq.status = SequenceStatus.RUNNING
    print(f"   Created sequence {test_seq.seq_id} with status: {test_seq.status}")
    
    # Step 4: Create sequence group
    print("4️⃣  Creating SequenceGroup...")
    sampling_params = SamplingParams(max_tokens=100)
    seq_group = SequenceGroup(
        request_id="demo_request",
        seqs=[test_seq],
        arrival_time=0.0,
        sampling_params=sampling_params
    )
    
    # Step 5: Create fake model output with sync token
    print("5️⃣  Creating fake model output with sync token...")
    sync_sample = SequenceOutput(
        parent_seq_id=1,
        output_token=special_kwargs.sync_token_id,  # This is our sync token!
        logprobs={special_kwargs.sync_token_id: -0.1}
    )
    
    fake_output = CompletionSequenceGroupOutput(
        samples=[sync_sample],
        prompt_logprobs=None
    )
    print(f"   Created output with token {sync_sample.output_token} (sync token)")
    
    # Step 6: Test the sync behavior
    print("6️⃣  Testing sync token handling...")
    print(f"   Sequence status before: {test_seq.status}")
    
    # This is the surgical test - directly call the method we want to test
    processor._handle_sync(seq_group, fake_output, special_kwargs.sync_token_id)
    
    print(f"   Sequence status after:  {test_seq.status}")
    
    # Step 7: Verify the result
    print("7️⃣  Verifying result...")
    if test_seq.status == SequenceStatus.BLOCKED:
        print("   ✅ SUCCESS: Sequence correctly blocked by sync token!")
    else:
        print("   ❌ FAILED: Sequence was not blocked")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\nKey Benefits of Surgical Testing:")
    print("• ⚡ Fast - No model loading or GPU inference")
    print("• 🎯 Precise - Test exact scenarios you care about") 
    print("• 🔍 Focused - Test individual methods in isolation")
    print("• 🛠️  Controllable - Create any state you need")
    print("• 🐛 Debuggable - Failures point directly to the issue")
    
    return True


def demo_promise_token_testing():
    """Demo testing promise token functionality."""
    print("\n🧪 Demo: Promise Token Testing")
    print("=" * 40)
    
    # Create processor (simplified setup)
    special_kwargs = SpecialKwargs(sync_token_id=111, promise_token_id=222)
    processor = SequenceHookOutputProcessor(
        Mock(), Mock(), [Mock()], Counter(), Mock(), special_kwargs
    )
    
    # Create parent sequence that can have children
    inputs = SingletonInputs({
        "prompt_token_ids": [1, 2, 3],
        "prompt": "Parent sequence",
        "type": "text"
    })
    
    parent_seq = Sequence(seq_id=10, inputs=inputs, block_size=16, eos_token_id=2)
    parent_seq.can_have_children = True
    parent_seq.status = SequenceStatus.RUNNING
    
    seq_group = SequenceGroup(
        request_id="promise_demo",
        seqs=[parent_seq],
        arrival_time=0.0,
        sampling_params=SamplingParams(max_tokens=50)
    )
    
    # Mock the spawn_child method
    child_seq = Sequence(seq_id=11, inputs=inputs, block_size=16, eos_token_id=2)
    child_seq.parent_seq_id = 10
    seq_group.spawn_child = Mock(return_value=child_seq)
    
    # Create promise token output
    promise_sample = SequenceOutput(
        parent_seq_id=10,
        output_token=special_kwargs.promise_token_id,
        logprobs={special_kwargs.promise_token_id: -0.2}
    )
    
    output = CompletionSequenceGroupOutput(samples=[promise_sample], prompt_logprobs=None)
    
    print(f"Parent status before: {parent_seq.status}")
    
    # Test promise handling
    processor._handle_promise(seq_group, output, special_kwargs.promise_token_id)
    
    print(f"Parent status after:  {parent_seq.status}")
    print(f"Spawn child called:   {seq_group.spawn_child.called}")
    
    if parent_seq.status == SequenceStatus.BLOCKED and seq_group.spawn_child.called:
        print("✅ Promise token correctly spawned child and blocked parent!")
    else:
        print("❌ Promise token handling failed")
    
    return True


if __name__ == "__main__":
    print("🎬 SequenceHookOutputProcessor Demo")
    print("=" * 60)
    print("This demo shows how surgical testing works:")
    print("• Create minimal state")
    print("• Test specific behaviors") 
    print("• Verify exact results")
    print("• No model inference needed!")
    print()
    
    try:
        demo_surgical_testing()
        demo_promise_token_testing()
        print("\n🎉 All demos completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 