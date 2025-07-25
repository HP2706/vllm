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
    Each request has its own instance of StreamingToolParser.
    """
    
    @classmethod
    def from_new_request(
        cls,
        tool_parser: ToolParser,
    ) -> "StreamingToolParser":
        return cls(tool_parser)
    
    def __init__(self, tool_parser: ToolParser):
        self.tool_parser = tool_parser
        self.chunks_mapping: List[DeltaMessage] = []  # chunks
        self.tool_states: Dict[int, Any] = {}  # tool_index -> state
        self.yielded_indices: Set[int] = set()  # yielded tool indices
        
        # Track accumulated text and token IDs per request
        self.accumulated_text: str = ""  # accumulated text
        self.accumulated_token_ids: List[int] = []  # accumulated token IDs
        
    def update_from_engine_output(
        self, 
        engine_output: EngineCoreOutput,
        tokenizer: AnyTokenizer,
    ) -> Optional[DeltaMessage]:
        """
        Process an EngineCoreOutput and extract tool call deltas.
        
        Args:
            engine_output: Output from the V1 engine core
            
        Returns:
            DeltaMessage containing tool call updates, if any
        """
        
        # Extract token information from engine output
        new_token_ids = engine_output.new_token_ids
        if not new_token_ids:
            return None
             
        # Decode the delta text from new tokens
        token_ids_list: List[int] = list(new_token_ids)
        delta_text: str = tokenizer.decode(token_ids_list, skip_special_tokens=True)
        
        # Get previous state
        previous_text = self.accumulated_text
        previous_token_ids = self.accumulated_token_ids.copy()
        
        # Update accumulated state
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + token_ids_list
        
        # Store updated state
        self.accumulated_text = current_text
        self.accumulated_token_ids = current_token_ids
        
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
            self.chunks_mapping.append(delta_message)
            
        return delta_message
        
    def get_tool_calls_from_deltas(self, delta: DeltaMessage) -> Optional[List[ToolCall]]:
        """
        Stream ToolCall objects as they become complete while processing DeltaMessage objects.
        
        Args:
            delta: DeltaMessage containing DeltaToolCall deltas
            
        Returns:
            List of complete ToolCall objects if any become ready, None otherwise
        """
        tool_state = self.tool_states
        yielded_indices = self.yielded_indices
    
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
    
    def process_engine_output_for_tools(
        self, 
        engine_output: EngineCoreOutput,
        tokenizer: AnyTokenizer,
    ) -> Optional[List[ToolCall]]:
        """
        Complete processing pipeline: extract deltas from engine output and return ready tool calls.
        
        Args:
            engine_output: Output from the V1 engine core
            
        Returns:
            List of complete ToolCall objects if any are ready
        """
        delta_message = self.update_from_engine_output(engine_output, tokenizer)
        if delta_message is None:
            return None
        
        return self.get_tool_calls_from_deltas(delta_message)
    
    def clear_state(self) -> None:
        """Clear state for a specific request."""
        self.chunks_mapping.clear()
        self.tool_states.clear()
        self.yielded_indices.clear()
        self.accumulated_text = ""
        self.accumulated_token_ids.clear()
        

