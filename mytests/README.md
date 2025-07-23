# SequenceHookOutputProcessor Test Suite

This directory contains comprehensive tests for the `SequenceHookOutputProcessor` functionality in vLLM. These tests are designed to be **surgical** - they directly manipulate vLLM's internal state to test specific behaviors without requiring full model inference.

## 🎯 What These Tests Cover

### Core Functionality Tests (`test_sequence_hook_processor.py`)
- **Sync Token Detection**: Tests that `<sync>` tokens properly block sequences
- **Promise Token Spawning**: Tests that `<promise>` tokens create child sequences  
- **Parent-Child Dependencies**: Tests dependency management and unblocking logic
- **Result Merging**: Tests merging child sequence results back into parents
- **Integration**: Tests full `process_outputs()` method integration

### Factory & Integration Tests (`test_processor_factory.py`)  
- **Factory Method**: Tests `SequenceGroupOutputProcessor.create_output_processor()`
- **Conditional Creation**: Tests proper processor selection based on `special_kwargs`
- **Inheritance**: Tests that `SequenceHookOutputProcessor` properly inherits
- **Configuration**: Tests integration with vLLM's configuration system

### Edge Cases & Error Handling (`test_edge_cases.py`)
- **Empty/Malformed Inputs**: Tests handling of empty outputs, None values, wrong types
- **Index Bounds**: Tests out-of-bounds sequence indices
- **Circular Dependencies**: Tests detection of circular parent-child relationships
- **Orphaned Children**: Tests handling of children with missing parents
- **Error Conditions**: Tests graceful handling of various error conditions

## 🚀 Running the Tests

### Run All Tests
```bash
# From the mytests directory
python run_all_tests.py
```

### Run Specific Test Suite
```bash
python run_all_tests.py processor    # Core processor tests
python run_all_tests.py factory      # Factory method tests  
python run_all_tests.py edge         # Edge case tests
```

### Run Individual Test Files
```bash
python test_sequence_hook_processor.py  # Core functionality
python test_processor_factory.py        # Factory methods
python test_edge_cases.py              # Edge cases
```

### Run with pytest (if available)
```bash
pytest test_*.py -v
```

## 🧪 Test Architecture

### Surgical Testing Approach

These tests use a **surgical approach** that directly manipulates vLLM's internal state:

1. **Mock Dependencies**: Create minimal mock objects for `Scheduler`, `Detokenizer`, etc.
2. **Direct State Manipulation**: Create `Sequence` and `SequenceGroup` objects with specific states
3. **Targeted Testing**: Test individual methods like `_handle_sync()`, `_handle_promise()`
4. **No Model Inference**: Tests run without loading actual models or running inference

### Key Test Utilities

```python
def create_test_sequence(seq_id: int, status: SequenceStatus):
    """Create a sequence with minimal required state."""
    
def create_test_sequence_group(request_id: str, seqs: List[Sequence]):
    """Create a sequence group with given sequences."""
    
def create_completion_output(samples: List[SequenceOutput]):
    """Create fake model outputs with specific tokens."""
```

## 📊 Example Test Flow

Here's how a typical test works:

```python
def test_sync_token_detection(self, processor):
    # 1. Create test sequence in RUNNING state
    seq = self.create_test_sequence(seq_id=1, status=SequenceStatus.RUNNING)
    seq_group = self.create_test_sequence_group("test_req", [seq])
    
    # 2. Create fake output with sync token
    sync_sample = self.create_sample_output(
        output_token=processor.special_kwargs.sync_token_id,
        parent_seq_id=1
    )
    output = self.create_completion_output([sync_sample])
    
    # 3. Test the behavior
    processor._handle_sync(seq_group, output, processor.special_kwargs.sync_token_id)
    
    # 4. Verify the result
    assert seq.status == SequenceStatus.BLOCKED
```

## 🔍 What Each Test Verifies

### Sync Token Tests
- ✅ Sequence status changes to `BLOCKED` when sync token detected
- ✅ Only sequences with sync tokens are affected
- ✅ Out-of-bounds sequence indices handled gracefully

### Promise Token Tests  
- ✅ Child sequences are spawned when promise token detected
- ✅ Parent sequence is blocked until children finish
- ✅ Spawn method is called with correct parameters
- ✅ Error handling for sequences that can't have children

### Dependency Management Tests
- ✅ Parents unblocked when all children finish
- ✅ Parents stay blocked if any children still running
- ✅ Child results properly merged into parents
- ✅ Circular dependencies detected without infinite loops

### Integration Tests
- ✅ `process_outputs()` calls sequence hook before standard processing
- ✅ Parent `SingleStepOutputProcessor.process_outputs()` still called
- ✅ Normal tokens don't trigger special behavior

## 🛠️ Extending the Tests

To add new tests:

1. **Add to existing files** for related functionality
2. **Create new test files** for major new features
3. **Update `run_all_tests.py`** to include new test suites
4. **Follow the surgical testing pattern** - create minimal state, test specific behavior

### Example New Test
```python
def test_new_functionality(self, processor):
    """Test description."""
    # 1. Setup minimal test state
    seq = self.create_test_sequence(...)
    
    # 2. Create specific test conditions
    output = self.create_completion_output(...)
    
    # 3. Test the behavior
    processor.new_method(...)
    
    # 4. Verify results
    assert expected_condition
    print("✅ New functionality test passed")
```

## 📈 Performance Considerations

The test suite includes basic performance testing to ensure the sequence hook logic doesn't add significant overhead:

- **Smoke Test**: Verifies basic imports and construction work
- **Performance Test**: Measures creation overhead (should be <1s for 100 instances)
- **Integration Test**: Ensures normal processing path still works efficiently

## 🎯 Benefits of This Testing Approach

1. **Fast Execution**: No model loading or GPU inference required
2. **Precise Control**: Test exactly the scenarios you care about
3. **Comprehensive Coverage**: Test edge cases that are hard to trigger naturally
4. **Easy Debugging**: Failures point directly to the problematic logic
5. **Maintainable**: Tests are independent and can run in any order

This surgical testing approach gives you confidence that your sequence hook implementation works correctly across all scenarios while maintaining fast development iteration cycles. 