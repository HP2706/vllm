# vLLM V1 Migration Analysis & Implementation Roadmap

## 🔄 Architecture Overview: V0 vs V1

vLLM V1 represents a fundamental architectural shift from V0, moving from a complex inheritance-based system to a cleaner, composition-based architecture optimized for performance and maintainability.

## 📊 Core Architectural Changes

### 1. **Request/Sequence Management**

#### V0 Architecture:
```python
# Complex hierarchy with multiple abstraction layers
SequenceGroup
├── List[Sequence]  # Multiple sequences per group
│   ├── SequenceData (token management)
│   ├── SequenceStatus (lifecycle state)
│   └── Block management
└── Sampling/Pooling params

# Usage in scheduler
scheduler.running: List[SequenceGroup]
seq_group.seqs: List[Sequence]  # Beam search, parallel sampling
```

#### V1 Architecture:
```python
# Simplified single Request object
Request
├── request_id: str
├── prompt_token_ids: List[int] 
├── output_token_ids: List[int] (private _output_token_ids)
├── RequestStatus (simplified enum)
├── sampling_params/pooling_params
└── Multimodal inputs

# Usage in scheduler  
scheduler.running: List[Request]
scheduler.requests: Dict[str, Request]  # Direct mapping
```

**Key Differences:**
- **V0**: Complex `SequenceGroup` → `Sequence` → `SequenceData` hierarchy
- **V1**: Flat `Request` object with direct token management
- **V0**: Supports beam search and complex forking within groups
- **V1**: Simpler model focused on single request processing
- **V0**: Block-level token management in `SequenceData`
- **V1**: Direct token arrays with `ConstantList` read-only views

### 2. **Output Processing Architecture**

#### V0 Output Processors:
```python
# Inheritance-based with multiple implementations
SequenceGroupOutputProcessor (ABC)
├── SingleStepOutputProcessor  # For beam search, single token/step
│   ├── process_outputs()     # Handles SequenceGroupOutput
│   ├── Detokenization
│   ├── Stop checking
│   └── Sequence forking/freeing
└── MultiStepOutputProcessor  # For spec decode, multi-token/step
    ├── Batch token processing
    ├── Rejection sampling
    └── No beam search support

# Factory pattern in interfaces.py
def create_output_processor():
    if num_lookahead_slots == 0:
        return SingleStepOutputProcessor()
    else:
        return MultiStepOutputProcessor()
```

#### V1 Output Processor:
```python
# Unified composition-based processor
OutputProcessor
├── RequestState management
├── Detokenization (per-request IncrementalDetokenizer)
├── Logprobs processing (LogprobsProcessor)
├── RequestOutputCollector (async queue replacement)
└── process_outputs(EngineCoreOutput) → RequestOutput

# Single implementation handles all cases
class OutputProcessor:
    def process_outputs(self, engine_core_outputs: List[EngineCoreOutput]):
        # Unified loop over all outputs
        for output in engine_core_outputs:
            # 1. Update stats
            # 2. Detokenize
            # 3. Create RequestOutput
            # 4. Handle queues vs return lists
```

**Key Differences:**
- **V0**: Multiple processor implementations based on scheduling mode
- **V1**: Single processor with unified logic
- **V0**: Complex `SequenceGroupOutput` handling with beam search
- **V1**: Simple `EngineCoreOutput` → `RequestOutput` transformation
- **V0**: Manual queue management for async operations
- **V1**: Built-in `RequestOutputCollector` for async handling

### 3. **Scheduler Architecture**

#### V0 Scheduler:
```python
# Phase-based scheduling
class Scheduler:
    waiting: Deque[SequenceGroup] 
    running: List[SequenceGroup]
    swapped: List[SequenceGroup]
    
    def schedule() -> SchedulerOutputs:
        # 1. Prefill phase - schedule waiting requests
        # 2. Decode phase - schedule running requests 
        # 3. Handle preemptions/swapping
        return SchedulerOutputs(
            scheduled_seq_groups=...,
            ignored_seq_groups=...,
            num_prefill_tokens=...,
            num_decode_tokens=...
        )
```

#### V1 Scheduler:
```python
# Unified token-based scheduling
class Scheduler(SchedulerInterface):
    requests: Dict[str, Request]
    waiting: RequestQueue  # FCFS or Priority policy
    running: List[Request]
    
    def schedule() -> SchedulerOutput:
        # Unified scheduling - no prefill/decode distinction
        # Just: {request_id: num_tokens} allocation
        token_budget = max_num_scheduled_tokens
        
        # Schedule running requests first
        # Then schedule waiting requests
        return SchedulerOutput(
            scheduled_new_reqs=...,
            scheduled_cached_reqs=..., 
            num_scheduled_tokens: Dict[str, int]
        )
```

**Key Differences:**
- **V0**: Phase-based (prefill vs decode) with complex state transitions
- **V1**: Unified token budget allocation regardless of phase
- **V0**: Complex preemption with swapping to CPU
- **V1**: Simplified preemption without swapping  
- **V0**: `SequenceGroupMetadata` for batch preparation
- **V1**: `NewRequestData`/`CachedRequestData` for worker communication

### 4. **Engine Structure**

#### V0 Engine:
```python
LLMEngine
├── Scheduler (handles sequence lifecycle)
├── ModelExecutor (model execution)
├── OutputProcessor (post-processing)
└── TokenizerGroup

# Monolithic design with tight coupling
```

#### V1 Engine:
```python
# Separated concerns with clear interfaces
LLMEngine / AsyncLLM
├── Processor (input processing)
├── OutputProcessor (output processing) 
├── EngineCore (core execution loop)
│   ├── Scheduler (request management)
│   ├── ModelExecutor (model execution)
│   └── StructuredOutputManager
└── EngineCoreClient (communication abstraction)

# Clean separation enables multiprocessing
```

## 🛣️ Implementation Roadmap: TODO.md Features for V1

The TODO.md outlines a sequence hook architecture for dependency-aware sequence lifecycle. Here's how to implement this in V1:

### Phase 1: Core V1 Hook Architecture 

#### 1.1 Enhanced Request Class for Dependencies
```python
# Extend vllm/v1/request.py
class Request:
    # Add dependency management fields
    parent_request_id: Optional[str] = None
    child_request_ids: Set[str] = field(default_factory=set)
    can_spawn_children: bool = False
    dependency_state: DependencyState = DependencyState.INDEPENDENT
    
    # Hook-related state
    detected_special_tokens: List[Tuple[int, SpecialTokenType]] = field(default_factory=list)
    pending_child_spawns: List[ChildSpawnConfig] = field(default_factory=list)

@dataclass 
class ChildSpawnConfig:
    token_position: int
    child_prompt: str
    child_sampling_params: SamplingParams
    spawn_after_token: bool = True  # spawn after generating the promise token
```

#### 1.2 V1 Output Processor Enhancement
```python
# Extend vllm/v1/engine/output_processor.py
class EnhancedOutputProcessor(OutputProcessor):
    """V1 Output processor with sequence hook capabilities."""
    
    def __init__(self, tokenizer: TokenizerGroup, log_stats: bool, 
                 special_token_config: Optional[SpecialTokenConfig] = None):
        super().__init__(tokenizer, log_stats)
        self.special_token_config = special_token_config
        self.dependency_manager = DependencyManager()
        
    def process_outputs(self, engine_core_outputs: List[EngineCoreOutput], 
                       **kwargs) -> OutputProcessorOutput:
        # Standard processing first
        result = super().process_outputs(engine_core_outputs, **kwargs)
        
        # Hook processing if special tokens configured
        if self.special_token_config:
            hook_requests = []
            for output in engine_core_outputs:
                if self._has_special_tokens(output):
                    hook_requests.append(output.request_id)
            
            # Process hooks
            spawn_requests, block_requests = self._process_hooks(hook_requests)
            result.spawn_requests = spawn_requests
            result.block_requests = block_requests
            
        return result
        
    def _process_hooks(self, request_ids: List[str]) -> Tuple[List[SpawnRequest], List[str]]:
        """Process special token hooks for sequence spawning/blocking."""
        spawn_requests = []
        block_requests = []
        
        for req_id in request_ids:
            request_state = self.request_states[req_id]
            
            # Check for sync tokens (blocking)
            if self._has_sync_token(request_state):
                block_requests.append(req_id)
                self.dependency_manager.block_request(req_id)
                
            # Check for promise tokens (spawning)
            if self._has_promise_token(request_state):
                spawn_config = self._extract_spawn_config(request_state)
                spawn_requests.append(SpawnRequest(
                    parent_id=req_id,
                    child_config=spawn_config
                ))
                
        return spawn_requests, block_requests
```

#### 1.3 V1 Scheduler Integration
```python
# Extend vllm/v1/core/sched/scheduler.py
class EnhancedScheduler(Scheduler):
    """V1 Scheduler with dependency awareness."""
    
    def __init__(self, vllm_config: VllmConfig, **kwargs):
        super().__init__(vllm_config, **kwargs)
        self.dependency_manager = DependencyManager()
        self.blocked_requests: Set[str] = set()
        
    def schedule(self) -> SchedulerOutput:
        # Standard V1 scheduling
        scheduler_output = super().schedule()
        
        # Filter out blocked requests from scheduling
        filtered_requests = self._filter_blocked_requests(scheduler_output)
        
        return scheduler_output._replace(
            scheduled_cached_reqs=filtered_requests
        )
        
    def add_child_request(self, parent_id: str, child_config: ChildSpawnConfig) -> str:
        """Spawn a child request from parent."""
        child_id = f"{parent_id}_child_{len(self.dependency_manager.get_children(parent_id))}"
        
        # Create child request
        parent_request = self.requests[parent_id]
        child_request = Request(
            request_id=child_id,
            prompt_token_ids=self._tokenize_prompt(child_config.child_prompt),
            sampling_params=child_config.child_sampling_params,
            parent_request_id=parent_id,
            # ... other fields from parent
        )
        
        # Register dependency
        self.dependency_manager.add_dependency(parent_id, child_id)
        self.blocked_requests.add(parent_id)  # Block parent until children finish
        
        # Add to scheduler
        self.add_request(child_request)
        return child_id
        
    def finish_request(self, request_id: str):
        """Override to handle dependency completion."""
        super().finish_request(request_id)
        
        # Check if this enables parent unblocking
        if parent_id := self.dependency_manager.get_parent(request_id):
            if self.dependency_manager.all_children_finished(parent_id):
                self.blocked_requests.discard(parent_id)
                # Optionally merge child results into parent
                self._merge_child_results(parent_id)
```

### Phase 2: Special Token Detection & Processing

#### 2.1 Token Stream Analysis
```python
# New file: vllm/v1/engine/special_token_processor.py
class SpecialTokenProcessor:
    """Handles detection and processing of special tokens in V1."""
    
    def __init__(self, special_token_config: SpecialTokenConfig):
        self.sync_token_id = special_token_config.sync_token_id
        self.promise_token_id = special_token_config.promise_token_id
        self.tokenizer = special_token_config.tokenizer
        
    def scan_for_special_tokens(self, new_token_ids: List[int], 
                               request_state: RequestState) -> List[SpecialTokenEvent]:
        """Scan new tokens for special token patterns."""
        events = []
        
        for i, token_id in enumerate(new_token_ids):
            if token_id == self.sync_token_id:
                events.append(SpecialTokenEvent(
                    type=SpecialTokenType.SYNC,
                    position=len(request_state.output_token_ids) + i,
                    request_id=request_state.request_id
                ))
            elif token_id == self.promise_token_id:
                # Extract promise payload from surrounding context
                payload = self._extract_promise_payload(new_token_ids, i, request_state)
                events.append(SpecialTokenEvent(
                    type=SpecialTokenType.PROMISE, 
                    position=len(request_state.output_token_ids) + i,
                    request_id=request_state.request_id,
                    payload=payload
                ))
                
        return events
        
    def _extract_promise_payload(self, tokens: List[int], promise_pos: int, 
                                request_state: RequestState) -> Dict[str, Any]:
        """Extract child request configuration from promise token context."""
        # Implementation depends on how promise payload is encoded
        # Could be JSON in subsequent tokens, special format, etc.
        pass
```

#### 2.2 Engine Core Integration
```python
# Modify vllm/v1/engine/core.py
class EngineCore:
    def __init__(self, vllm_config: VllmConfig, **kwargs):
        super().__init__(vllm_config, **kwargs)
        
        # Initialize special token processing if configured
        self.special_token_processor = None
        if hasattr(vllm_config, 'special_token_config'):
            self.special_token_processor = SpecialTokenProcessor(
                vllm_config.special_token_config
            )
            
        # Use enhanced scheduler if special tokens enabled
        if self.special_token_processor:
            self.scheduler = EnhancedScheduler(vllm_config, **kwargs)
        
    def step(self) -> Tuple[Dict[int, EngineCoreOutputs], bool]:
        """Enhanced step with hook processing."""
        outputs, executed = super().step()
        
        # Process hooks after standard execution
        if self.special_token_processor and outputs:
            self._process_special_token_hooks(outputs)
            
        return outputs, executed
        
    def _process_special_token_hooks(self, outputs: Dict[int, EngineCoreOutputs]):
        """Process special token events from this step."""
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                if special_events := self._detect_special_tokens(output):
                    self._handle_special_events(special_events)
                    
    def _handle_special_events(self, events: List[SpecialTokenEvent]):
        """Handle detected special token events."""
        for event in events:
            if event.type == SpecialTokenType.SYNC:
                self.scheduler.block_request(event.request_id)
            elif event.type == SpecialTokenType.PROMISE:
                child_config = self._create_child_config(event)
                self.scheduler.add_child_request(event.request_id, child_config)
```

### Phase 3: Advanced Features & Integration

#### 3.1 Result Merging & Context Injection
```python
# Enhancement to dependency manager
class DependencyManager:
    def merge_child_results(self, parent_id: str) -> str:
        """Merge child results for parent continuation."""
        children = self.get_completed_children(parent_id)
        child_outputs = [self.get_request_output(child_id) for child_id in children]
        
        # Simple concatenation strategy (can be made configurable)
        merged_text = "\n".join(output.text for output in child_outputs)
        
        # Could also inject as additional context tokens
        return merged_text
        
    def inject_child_context(self, parent_request: Request, merged_results: str):
        """Inject child results as context for parent continuation.""" 
        # Option 1: Modify parent's prompt_token_ids
        additional_tokens = self.tokenizer.encode(merged_results)
        parent_request.prompt_token_ids.extend(additional_tokens)
        
        # Option 2: Use KV cache injection (more advanced)
        # Would require deeper integration with KV cache manager
```

#### 3.2 Configuration Integration
```python
# Extend vllm/config.py with special token config
@dataclass
class SpecialTokenConfig:
    """Configuration for special token processing."""
    sync_token_id: Optional[int] = None
    promise_token_id: Optional[int] = None
    enable_dependency_management: bool = False
    child_merge_strategy: str = "concatenate"  # "concatenate", "kv_inject", "none"
    max_child_depth: int = 1  # Prevent infinite recursion

# Extend VllmConfig
@dataclass  
class VllmConfig:
    # ... existing fields
    special_token_config: Optional[SpecialTokenConfig] = None
```

#### 3.3 API Integration
```python
# Extend vllm/v1/engine/async_llm.py
class AsyncLLM(EngineClient):
    def __init__(self, vllm_config: VllmConfig, **kwargs):
        super().__init__(vllm_config, **kwargs)
        
        # Initialize enhanced output processor if special tokens configured
        if vllm_config.special_token_config:
            self.output_processor = EnhancedOutputProcessor(
                self.tokenizer,
                log_stats=self.log_stats,
                special_token_config=vllm_config.special_token_config
            )
```

## 🔧 Implementation Strategy

### Minimal Invasive Approach
1. **Start with OutputProcessor**: Extend the existing V1 `OutputProcessor` rather than creating entirely new classes
2. **Leverage RequestState**: Use existing `RequestState` management but add dependency tracking
3. **Enhance Scheduler Gradually**: Start with basic blocking/unblocking, add spawning incrementally
4. **Configuration-Driven**: Make all special token features opt-in via configuration

### Compatibility Considerations
- **V1 API Preservation**: All changes should be backward compatible with standard V1 usage
- **Performance**: Special token processing should have zero overhead when disabled
- **Multiprocessing**: Ensure dependency management works across V1's multiprocess architecture

## 📈 Benefits of V1 Implementation

1. **Cleaner Architecture**: V1's composition-based design makes hook integration more natural
2. **Better Performance**: Unified token scheduling reduces complexity vs V0's phase-based approach
3. **Simpler State Management**: Single `Request` objects easier to manage than `SequenceGroup` hierarchies  
4. **Native Multiprocessing**: V1's architecture better supports distributed dependency management
5. **Future-Proof**: Aligns with vLLM's architectural direction and deprecation of V0

## 🎯 Testing Strategy

1. **Unit Tests**: Test special token detection and dependency management in isolation
2. **Integration Tests**: Test end-to-end hook workflows with actual model inference
3. **Performance Tests**: Ensure zero overhead when hooks disabled
4. **Surgical Testing**: Continue using the surgical testing approach from TODO.md for rapid iteration

This roadmap leverages V1's cleaner architecture to implement the sequence hook functionality more elegantly than would be possible in V0's complex inheritance-based system. 