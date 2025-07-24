# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Dict, List, Set, Optional
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.protocol import DeltaMessage, ToolCall, FunctionCall
from . import EngineCoreOutput 

logger = init_logger(__name__)

class StreamingToolParser:
    """
    V1 Streaming tool parser that integrates with the V1 OutputProcessor architecture.
    Handles streaming tool parsing for EngineCoreOutput and maintains state per request.
    """
    
    def __init__(self, anytokenizer: AnyTokenizer, tool_parser: ToolParser):
        self.anytokenizer = anytokenizer
        self.tool_parser = tool_parser
        self.chunks_mapping: Dict[str, List[DeltaMessage]] = {}  # request_id -> chunks
        self.tool_states: Dict[str, Dict[int, Any]] = {}  # request_id -> tool_index -> state
        self.yielded_indices: Dict[str, Set[int]] = {}  # request_id -> yielded tool indices
        
        # Track accumulated text and token IDs per request
        self.accumulated_text: Dict[str, str] = {}  # request_id -> accumulated text
        self.accumulated_token_ids: Dict[str, List[int]] = {}  # request_id -> accumulated token IDs
        
    def _initialize_request_state(self, request_id: str) -> None:
        """Initialize state for a new request if not already present."""
        if request_id not in self.chunks_mapping:
            self.chunks_mapping[request_id] = []
        if request_id not in self.tool_states:
            self.tool_states[request_id] = {}
        if request_id not in self.yielded_indices:
            self.yielded_indices[request_id] = set()
        if request_id not in self.accumulated_text:
            self.accumulated_text[request_id] = ""
        if request_id not in self.accumulated_token_ids:
            self.accumulated_token_ids[request_id] = []
        
    def update_from_engine_output(self, engine_output: EngineCoreOutput) -> Optional[DeltaMessage]:
        """
        Process an EngineCoreOutput and extract tool call deltas.
        
        Args:
            engine_output: Output from the V1 engine core
            
        Returns:
            DeltaMessage containing tool call updates, if any
        """
        request_id = engine_output.request_id
        self._initialize_request_state(request_id)
        
        # Extract token information from engine output
        new_token_ids = engine_output.new_token_ids
        if not new_token_ids:
            return None
             
        # Decode the delta text from new tokens
        token_ids_list: List[int] = list(new_token_ids)
        delta_text: str = self.anytokenizer.decode(token_ids_list, skip_special_tokens=True)
        
        # Get previous state
        previous_text = self.accumulated_text[request_id]
        previous_token_ids = self.accumulated_token_ids[request_id].copy()
        
        # Update accumulated state
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + token_ids_list
        
        # Store updated state
        self.accumulated_text[request_id] = current_text
        self.accumulated_token_ids[request_id] = current_token_ids
        
        # Call the existing tool parser with proper accumulated context
        try:
            delta_message = self.tool_parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=token_ids_list,
                request=None  # type: ignore - V1 might have request context available
            )
        except TypeError:
            # Handle case where tool parser doesn't accept None for request
            delta_message = None
        
        if delta_message is not None:
            self.chunks_mapping[request_id].append(delta_message)
            
        return delta_message
        
    def get_tool_calls_from_deltas(self, request_id: str, delta: DeltaMessage) -> Optional[List[ToolCall]]:
        """
        Stream ToolCall objects as they become complete while processing DeltaMessage objects.
        
        Args:
            request_id: Request ID to process deltas for
            delta: DeltaMessage containing DeltaToolCall deltas
            
        Returns:
            List of complete ToolCall objects if any become ready, None otherwise
        """
        # Ensure request state is initialized
        self._initialize_request_state(request_id)
        
        tool_state = self.tool_states[request_id]
        yielded_indices = self.yielded_indices[request_id]
    
        tool_calls = []
        if not delta.tool_calls:
            return None
        
        for tool_call in delta.tool_calls:
            index = tool_call.index
      
            # Initialize state for new tool call
            if index not in tool_state:
                tool_state[index] = {
                    'id': None,
                    'type': None,
                    'function_name': None,
                    'arguments': "",
                    'brace_count': 0  # Track { and } brackets
                }
            
            state = tool_state[index]
            
            # Update state with new information
            if tool_call.id is not None:
                state['id'] = tool_call.id
            if tool_call.type is not None:
                state['type'] = tool_call.type
            if tool_call.function is not None:
                if tool_call.function.name is not None:
                    state['function_name'] = tool_call.function.name
                if tool_call.function.arguments is not None:
                    new_args = tool_call.function.arguments
                    state['arguments'] += new_args
                    # Track brackets to determine when arguments are complete
                    for char in new_args:
                        if char == '{':
                            state['brace_count'] += 1
                        elif char == '}':
                            state['brace_count'] -= 1
            
            # Check if tool call becomes ready for the first time
            # Arguments are complete when brace_count is 0 and we have some arguments
            arguments_complete = (state['brace_count'] == 0 and 
                                state['arguments'] != "" and 
                                '{' in state['arguments'])
            
            if (index not in yielded_indices and
                state['id'] is not None and 
                state['type'] is not None and 
                state['function_name'] is not None and
                arguments_complete):
                
                # Yield immediately when ready!
                yielded_indices.add(index)
                tool_calls.append(ToolCall(
                    id=state['id'],
                    type=state['type'],
                    function=FunctionCall(
                        name=state['function_name'],
                        arguments=state['arguments']
                    )
                ))
        
        return tool_calls if tool_calls else None
    
    def process_engine_output_for_tools(self, engine_output: EngineCoreOutput) -> Optional[List[ToolCall]]:
        """
        Complete processing pipeline: extract deltas from engine output and return ready tool calls.
        
        Args:
            engine_output: Output from the V1 engine core
            
        Returns:
            List of complete ToolCall objects if any are ready
        """
        delta_message = self.update_from_engine_output(engine_output)
        if delta_message is None:
            return None
        
        return self.get_tool_calls_from_deltas(engine_output.request_id, delta_message)
    
    def clear_request_state(self, request_id: str) -> None:
        """Clear state for a specific request."""
        self.chunks_mapping.pop(request_id, None)
        self.tool_states.pop(request_id, None)
        self.yielded_indices.pop(request_id, None)
        self.accumulated_text.pop(request_id, None)
        self.accumulated_token_ids.pop(request_id, None)
        
    def get_request_chunks(self, request_id: str) -> List[DeltaMessage]:
        """Get all accumulated chunks for a request."""
        return self.chunks_mapping.get(request_id, [])


class ToolAwareRequestState:
    """
    Enhanced RequestState that includes tool parsing capabilities.
    This would integrate with the V1 OutputProcessor architecture.
    """
    
    def __init__(self, request_id: str, tool_parser: Optional[StreamingToolParser] = None):
        self.request_id = request_id
        self.tool_parser = tool_parser
        self.accumulated_tool_calls: List[ToolCall] = []
        
    def process_engine_output(self, engine_output: EngineCoreOutput) -> Optional[List[ToolCall]]:
        """Process engine output and return any new complete tool calls."""
        if self.tool_parser is None:
            return None
            
        new_tool_calls = self.tool_parser.process_engine_output_for_tools(engine_output)
        if new_tool_calls:
            self.accumulated_tool_calls.extend(new_tool_calls)
            
        return new_tool_calls
        
    def get_all_tool_calls(self) -> List[ToolCall]:
        """Get all tool calls accumulated for this request."""
        return self.accumulated_tool_calls.copy()
        
    def clear_tool_state(self) -> None:
        """Clear tool parsing state for this request."""
        if self.tool_parser:
            self.tool_parser.clear_request_state(self.request_id)
        self.accumulated_tool_calls.clear()
