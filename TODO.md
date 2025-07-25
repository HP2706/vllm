# ✅ Tool-Call Based Sequence Spawning for vLLM v1

This document outlines the **detailed implementation plan** for tool-call based sequence spawning in vLLM v1. The goal is to implement `spawn_child()`, `sync()`, and `return_to_parent()` functions that allow sequences to spawn children, manage dependencies, and resume efficiently using vLLM v1's Request-based architecture.

## 🎯 **Focused Implementation Tasks**

### 1. ToolCallExecutor Class
**File**: `vllm/v1/engine/output_processor.py`
- [ ] **An abstraction for executing arbitrary tool calls**
- [ ] **These tool calls will impact RequestState and should update the engine core**


### 2. Modifications needed in engine core to create new requests from existing ones 
**File**: `vllm/v1/engine/core.py`
- [ ] **Implement Request spawning mechanism** (`v1-request-spawning`)

```python
class EngineCore:
    def __init__(self, ...):
        # ... existing code ...
        self.dependency_manager = DependencyManager()
    
    def spawn_request(self, parent_request: Request, child_prompt: str) -> str:
        """Create child request with parent context"""
        child_id = f"{parent_request.request_id}_child_{uuid4()}"
        
        # Inherit context from parent
        child_request = Request(
            request_id=child_id,
            prompt_token_ids=self._tokenize_prompt(child_prompt),
            sampling_params=parent_request.sampling_params.clone(),
            eos_token_id=parent_request.eos_token_id,
            parent_id=parent_request.request_id,  # NEW FIELD
            priority=parent_request.priority + 1,  # Higher priority for children
        )
        
        # Register dependency
        self.dependency_manager.register_dependency(parent_request.request_id, child_id)
        
        # Add to scheduler
        self.scheduler.add_request(child_request)
        return child_id
```

### 3. Dependency Management System
**File**: `vllm/v1/core/dependency_manager.py` 
- [ ] **Create cross-Request dependency tracker** (`v1-dependency-tracker`)

```python
@dataclass
class DependencyInfo:
    parent_id: str
    child_ids: list[str] 
    completed_children: set[str]
    child_results: dict[str, str]  # child_id -> result_text
    sync_waiting: bool = False

class DependencyManager:
    def __init__(self):
        self.dependencies: dict[str, DependencyInfo] = {}  # parent_id -> info
        self.child_to_parent: dict[str, str] = {}  # child_id -> parent_id
    
    def register_dependency(self, parent_id: str, child_id: str) -> None:
        """Register parent-child relationship"""
        if parent_id not in self.dependencies:
            self.dependencies[parent_id] = DependencyInfo(parent_id, [], set(), {})
        
        self.dependencies[parent_id].child_ids.append(child_id)
        self.child_to_parent[child_id] = parent_id
    
    def mark_child_complete(self, child_id: str, result: str) -> Optional[str]:
        """Mark child complete and return parent_id if all children done"""
        if child_id not in self.child_to_parent:
            return None
            
        parent_id = self.child_to_parent[child_id]
        dep_info = self.dependencies[parent_id]
        
        dep_info.completed_children.add(child_id)
        dep_info.child_results[child_id] = result
        
        # Check if parent can resume
        if (dep_info.sync_waiting and 
            len(dep_info.completed_children) >= len(dep_info.child_ids)):
            return parent_id
        return None
        
    def get_child_results(self, parent_id: str, child_ids: list[str] = None) -> list[str]:
        """Get results from specified children (or all if None)"""
        if parent_id not in self.dependencies:
            return []
            
        dep_info = self.dependencies[parent_id]
        target_ids = child_ids or dep_info.child_ids
        
        return [dep_info.child_results.get(cid, "") for cid in target_ids]
```

### 4. OutputProcessor Integration
**File**: `vllm/v1/engine/output_processor.py`
- [ ] **Integrate tool call detection with request spawning** (`v1-output-processor-integration`)

```python
class OutputProcessor:
    def __init__(self, tokenizer: TokenizerGroup, log_stats: bool):
        # ... existing code ...
        self.spawning_parser = SequenceSpawningToolParser(tokenizer)
        self.dependency_manager = None  # Set by EngineCore
    
    def process_outputs(self, engine_core_outputs: list[EngineCoreOutput], ...):
        # ... existing code ...
        
        for output in engine_core_outputs:
            # Check for tool calls
            tool_calls = self.spawning_parser.process_engine_output_for_tools(output)
            
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.function.name == "spawn_child":
                        args = json.loads(tool_call.function.arguments)
                        child_id = self._handle_spawn_request(output.request_id, args["prompt"])
                        
                    elif tool_call.function.name == "sync":
                        args = json.loads(tool_call.function.arguments)
                        self._handle_sync_request(output.request_id, args.get("child_ids"))
                        
                    elif tool_call.function.name == "return_to_parent":
                        args = json.loads(tool_call.function.arguments) 
                        self._handle_return_request(output.request_id, args["text"])
```

### 5. Scheduler Dependency Handling
**File**: `vllm/v1/core/sched/scheduler.py`
- [ ] **Add dependency-aware scheduling** (`v1-scheduler-deps`)

```python
class Scheduler(SchedulerInterface):
    def __init__(self, ...):
        # ... existing code ...
        self.dependency_manager = None  # Set by EngineCore
        self.blocked_requests: set[str] = set()  # Requests waiting for children
    
    def schedule(self) -> SchedulerOutput:
        # ... existing code ...
        
        # Filter out blocked requests waiting for children
        available_requests = [
            req for req in self.waiting_requests 
            if req.request_id not in self.blocked_requests
        ]
        
        # Prioritize child requests (higher priority values)
        available_requests.sort(key=lambda r: r.priority, reverse=True)
        
        # ... continue with existing scheduling logic ...
        
    def block_request_for_sync(self, request_id: str) -> None:
        """Block parent request until children complete"""
        self.blocked_requests.add(request_id)
        
    def unblock_request(self, request_id: str) -> None:
        """Unblock parent request when children complete"""
        self.blocked_requests.discard(request_id)
```

### 6. RequestOutput update to include tool call history, when it was called and result, children history etc.

### 6. Comprehensive Testing
**Directory**: `tests/v1/sequence_spawning/`
- [ ] **Create complete test suite** (`v1-testing-framework`)

```python
# tests/v1/sequence_spawning/test_request_spawning.py
class TestRequestSpawning:
    def test_spawn_child_creates_dependency(self):
        """Test that spawn_child creates proper parent-child relationship"""
        
    def test_sync_blocks_parent_until_children_complete(self):
        """Test sync mechanism blocks parent appropriately"""
        
    def test_return_to_parent_merges_results(self):
        """Test child results are properly merged into parent"""
        
    def test_circular_dependency_detection(self):
        """Test prevention of circular dependencies"""
        
    def test_child_failure_handling(self):
        """Test graceful handling when child requests fail"""

# tests/v1/sequence_spawning/test_dependency_manager.py  
class TestDependencyManager:
    def test_register_multiple_children(self):
        """Test registering multiple children for same parent"""
        
    def test_partial_completion_handling(self):
        """Test behavior with only some children completed"""
```

## 🔄 **Implementation Flow**

### **Phase 1: Core Infrastructure** 
1. `v1-spawn-functions` - Tool function definitions and parser
2. `v1-dependency-tracker` - Dependency management system

### **Phase 2: Engine Integration**
3. `v1-request-spawning` - Request creation in EngineCore  
4. `v1-output-processor-integration` - Tool call detection

### **Phase 3: Scheduling & Testing**
5. `v1-scheduler-deps` - Scheduler modifications
6. `v1-testing-framework` - Comprehensive test coverage

## 📋 **Key Integration Points**

- **Tool Parser**: Extends existing `StreamingToolParser` architecture
- **Request Objects**: Adds `parent_id` field to `Request` class  
- **Engine Pipeline**: Integrates at `OutputProcessor` level for clean separation
- **Scheduler**: Minimal changes with dependency-aware request filtering
- **Memory Sharing**: Leverage existing KV cache sharing for parent-child context

## 🚀 **Usage Example**

```python
# Model can now call these functions:
child_id = spawn_child("Analyze this data in detail: {...}")
results = sync([child_id])  # Waits for child completion  
return_to_parent(f"Analysis: {results[0]}")
```

