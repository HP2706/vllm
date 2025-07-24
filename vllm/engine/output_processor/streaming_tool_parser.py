# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Dict, List, Set
from vllm.logger import init_logger
from vllm.sequence import (SequenceGroup)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.protocol import DeltaMessage, ToolCall, FunctionCall

logger = init_logger(__name__)

class StreamingToolParser:
    """
    Handles streaming tool parsing for sequence groups and individual sequences.
    Maintains state per sequence to properly track tool call progress across multiple tokens.
    """
    
    def __init__(self, anytokenizer: AnyTokenizer, tool_parser: ToolParser):
        self.anytokenizer = anytokenizer
        self.tool_parser = tool_parser
        self.chunks_mapping: Dict[int, List[DeltaMessage]] = {}  # we maintain chunks for each sequence group
        self.is_complete = False
        self.tool_states: Dict[int, Dict[int, Any]] = {}  # Changed from Dict[str, Any] to Dict[int, Any] for tool indices
        self.yielded_indices: Dict[int, Set[int]] = {}
        
    def _initialize_seq_state(self, seq_id: int) -> None:
        """Initialize state for a new sequence if not already present."""
        if seq_id not in self.chunks_mapping:
            self.chunks_mapping[seq_id] = []
        if seq_id not in self.tool_states:
            self.tool_states[seq_id] = {}
        if seq_id not in self.yielded_indices:
            self.yielded_indices[seq_id] = set()
        
    def update(self, seq_group: SequenceGroup) -> None:
        assert len(seq_group.seqs) == 1, f'We only support single element sequence groups'
        seq_id = seq_group.seqs[0].seq_id
        self._initialize_seq_state(seq_id)
        
        delta_message = self.get_delta(seq_group)
        if delta_message is not None:
            self.chunks_mapping[seq_id].append(delta_message)
        
    def get_delta(self, seq_group: SequenceGroup) -> DeltaMessage | None:
        """
        Get the delta message using the existing tool parser
        this allows us to use any tool parser that implements the extract_tool_calls_streaming method
        We assume the sequence group has been updated 
        """
        assert len(seq_group.seqs) == 1, f'We only support single element sequence groups'
        seq = seq_group.seqs[0]  # Process first sequence
        tool_parser = self.tool_parser
        
        delta = seq.data.get_delta_and_reset()
        
        current_token_ids = seq.data.get_token_ids()
        delta_token_ids = delta.new_output_token_ids
        previous_token_ids = current_token_ids[:-len(delta_token_ids)] # old token ids = current token ids - delta token ids
        
        # Get current token state
        current_text = seq.output_text
        delta_text = self.anytokenizer.decode(delta.new_output_token_ids)
        previous_text = current_text[:-len(delta_text)] # old text = current text - delta text
        
        # Call streaming parser with actual request
        return tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=None  # type: ignore - Not used by most parsers
        )
        
    def get_tool_calls_from_deltas(self, seq_id: int, delta: DeltaMessage) -> List[ToolCall] | None:
        """
        Stream ToolCall objects as they become complete while processing DeltaMessage objects.
        Yields ToolCall objects as soon as they have all required information (id, type, function name),
        with accumulated arguments up to that point. This is truly streaming - tool calls are yielded
        immediately when they become ready, not batched at the end.
        
        Args:
            seq_id: Sequence ID to process deltas for
            delta: DeltaMessage containing DeltaToolCall deltas
            
        Returns:
            ToolCall: Complete tool call if one becomes ready, None otherwise
        """
        # Ensure seq_id state is initialized
        self._initialize_seq_state(seq_id)
        
        tool_state = self.tool_states[seq_id]
        yielded_indices = self.yielded_indices[seq_id]
    
        tool_calls = []
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
            
            state = tool_state[index]  # Fix: define state variable properly
            
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
                                state['arguments'] and 
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
        
        if len(tool_calls) > 0:
            return tool_calls
        else:
            return None
    
    def clear_seq_state(self, seq_id: int) -> None:
        """Clear state for a specific sequence."""
        self.chunks_mapping.pop(seq_id, None)
        self.tool_states.pop(seq_id, None)
        self.yielded_indices.pop(seq_id, None)
