# vLLM Sequence Generation Overview

This document provides a detailed overview of the sequence generation process in vLLM, from the user-facing `llm.generate` call to the low-level execution and resource management. It also covers the **sequence hook architecture** for advanced sequence dependency management.

## 1. Entrypoint: `llm.generate`

The sequence generation process begins with a call to the `llm.generate` method in `vllm/entrypoints/llm.py`. This method is designed for offline inference.

The `llm.generate` method performs the following key steps:

1.  **Request Validation and Preparation**: It calls the internal `_validate_and_add_requests` method. This method iterates through the input prompts, assigning a unique `request_id` to each.
2.  **Adding Requests to the Engine**: For each prompt, `_validate_and_add_requests` calls `_add_request`, which in turn calls `self.llm_engine.add_request`. This action passes the request from the `LLM` entrypoint to the `LLMEngine` for processing.

## 2. The `LLMEngine`: Adding a Request

The `LLMEngine` is the core component that orchestrates the entire sequence generation process. When `add_request` is called on the `LLMEngine` in `vllm/engine/llm_engine.py`, it performs the following:

1.  **Input Preprocessing**: The `add_request` method calls `self.input_preprocessor.preprocess`. This function tokenizes the input prompt and prepares the necessary model inputs.
2.  **Sequence and SequenceGroup Creation**: A `Sequence` object is created to represent a single generation sequence. A `SequenceGroup` is created to manage one or more sequences that originate from the same prompt. This is important for techniques like beam search.
3.  **Adding to the Scheduler**: The `SequenceGroup` is then added to the scheduler via `scheduler.add_seq_group(seq_group)`. The scheduler's job is to manage the order in which `SequenceGroup`s are processed by the model.

## 3. The `step` Method: Driving the Generation

The generation process is driven by the `llm_engine.step()` method, which is called in a loop. Each call to `step()` represents one iteration of the generation process. Here's what happens inside:

1.  **Scheduling**: The `scheduler.schedule()` method is called to select the next batch of `SequenceGroup`s to run. The scheduler implements the chosen scheduling policy (e.g., FCFS, PagedAttention) and manages the KV cache to prevent out-of-memory errors.
2.  **Model Execution**: The selected `SequenceGroup`s are passed to `model_executor.execute_model()`. This triggers a forward pass of the model on the GPU, which computes the logits for the next token in each sequence.
3.  **Processing Outputs**: The model's output (the logits) is processed. This involves:
    *   **Sampling**: Applying the specified sampling parameters to the logits to select the next token for each sequence.
    *   **Sequence Hook Processing**: If special tokens are configured, the `SequenceHookOutputProcessor` checks for special tokens (`<sync>`, `<promise>`) and manages sequence dependencies (blocking/unblocking parents, spawning children).
    *   **Updating Sequences**: Appending the newly generated tokens to their respective sequences.
    *   **Checking for Completion**: Identifying sequences that have finished generation (e.g., by reaching the maximum length or generating an EOS token).
4.  **Resource Deallocation**: When a `SequenceGroup` is finished, its resources, particularly the KV cache blocks it was using, are freed.
5.  **Returning Results**: The `step()` method returns the generated outputs for any requests that have completed in the current iteration.

## 4. Resource Deallocation: Freeing the KV Cache

A key aspect of vLLM's performance is its efficient management of the KV cache. When a sequence finishes generation, the resources it consumed must be released so they can be used by other requests. This process is handled as follows:

1.  **Identifying Finished Sequences**: In each `step`, the engine identifies `SequenceGroup`s that have finished generating text.
2.  **Freeing the Sequence**: For each finished `SequenceGroup`, the scheduler's `free_finished_seq_groups` method is called. This method, in turn, calls `_free_finished_seq_group` which calls `free_seq` for each finished sequence within the group.
3.  **Releasing KV Cache Blocks**: The `free_seq` method then calls `self.block_manager.free(seq)`. This is the final step where the `BlockManager` releases the physical KV cache blocks that were allocated to the sequence, making them available for new requests.

This lifecycle ensures that the KV cache is used efficiently, allowing vLLM to achieve high throughput.

## 5. Sequence Hook Architecture: Advanced Dependency Management

vLLM includes a sophisticated **sequence hook architecture** that enables advanced sequence dependency management through special tokens and output processors. This system allows sequences to dynamically spawn children, manage dependencies, and coordinate complex generation workflows.

### 5.1 Special Token Processing

When special tokens are configured via `SpecialKwargs`, vLLM automatically uses the `SequenceHookOutputProcessor` instead of the standard `SingleStepOutputProcessor`. This processor detects special tokens in the model output and triggers corresponding sequence management actions:

*   **`<sync>` Token**: When detected, the current sequence is paused (status set to `BLOCKED`) to allow for coordination with other sequences.
*   **`<promise>` Token**: Triggers the spawning of a child sequence with a dependency relationship to the parent.

### 5.2 Parent-Child Dependencies

The sequence hook system implements a robust parent-child dependency model:

1.  **Child Spawning**: When a `<promise>` token is detected, the system spawns a new child sequence from the parent.
2.  **Parent Blocking**: The parent sequence is automatically blocked (`BLOCKED` status) until all its children complete.
3.  **Dependency Tracking**: The system maintains parent-child relationships and monitors child completion status.
4.  **Automatic Unblocking**: When all children finish, the parent is automatically unblocked and set to `WAITING` status.

### 5.3 Result Merging and Conditional Prefill

A key performance feature is the efficient merging of child results back into parent sequences:

1.  **Result Collection**: When children finish, their output text is collected.
2.  **Merge Strategy**: Child outputs are merged using a configurable strategy (default: newline-separated concatenation).
3.  **Context Appending**: Merged results are appended to the parent sequence's context.
4.  **Conditional Prefill**: The scheduler automatically processes only the new tokens on the next cycle, leveraging existing KV cache for efficiency.

### 5.4 Integration with Output Processing Pipeline

The sequence hook architecture integrates seamlessly with vLLM's existing output processing pipeline:

```
Model Output → Sampling → Sequence Hook Processing → Detokenization → Stop Checking
```

The `SequenceHookOutputProcessor` runs **before** detokenization and stop checking, ensuring that:
- Special tokens are processed before being converted to text
- Sequence dependencies are managed at the token level
- Standard output processing continues normally for non-special tokens

### 5.5 Configuration and Usage

```python
from vllm import LLM, SamplingParams
from vllm.config import SpecialKwargs

# Configure special tokens
special_kwargs = SpecialKwargs(
    sync_token_id=12345,    # <sync> token ID  
    promise_token_id=67890, # <promise> token ID
)

# Initialize with sequence hooks enabled
llm = LLM(
    model="your-model-path",
    special_kwargs=special_kwargs  # Enables SequenceHookOutputProcessor
)

# Generate - sequence hooks work automatically
outputs = llm.generate(prompts, sampling_params)
```

### 5.6 Benefits and Use Cases

The sequence hook architecture enables several advanced use cases:

*   **Multi-Step Reasoning**: Sequences can spawn children for intermediate reasoning steps
*   **Parallel Processing**: Related computations can run in parallel before merging results  
*   **Dynamic Workflows**: Generation can adapt based on intermediate results
*   **Efficient Context Management**: Conditional prefill minimizes re-computation overhead

This architecture provides a powerful foundation for building sophisticated AI reasoning and generation workflows while maintaining vLLM's high-performance characteristics. 