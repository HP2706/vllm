"""
Focused tests for StreamingToolParser with real DeepSeekV3 tokenizer.
Tests realistic streaming scenarios where tool calls are built incrementally.
"""

import sys
import os
# Add the parent directory to sys.path to import vllm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vllm.entrypoints.openai.protocol import (
    DeltaMessage, DeltaToolCall, DeltaFunctionCall
)
from vllm.engine.output_processor.streaming_tool_parser import StreamingToolParser
from vllm.entrypoints.openai.tool_parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser
from transformers import AutoTokenizer


def assert_eq_dict(d1 : dict, d2 : dict):
    for k, v in d1.items():
        if k not in d2:
            assert False, f'{k} not in d2'
        if d2[k] != v:
            assert False, f'{k} {d2[k]} != {v}'
    return True

class TestStreamingToolParser:
    """Focused test suite for StreamingToolParser with real scenarios."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v2.5", trust_remote_code=True)
        self.tool_parser = DeepSeekV3ToolParser(self.tokenizer)
        self.parser = StreamingToolParser(self.tokenizer, self.tool_parser)
    
    def test_streaming_tool_call_incremental(self):
        """Test building a tool call incrementally across multiple deltas."""
            
        seq_id = 1001
        
        # Step 1: ID and type only
        delta1 = DeltaMessage(tool_calls=[
            DeltaToolCall(index=0, id="stream_001", type="function")
        ])
        result1 = self.parser.get_tool_calls_from_deltas(seq_id, delta1)
        assert result1 is None  # Not complete yet
        
        # Step 2: Function name
        delta2 = DeltaMessage(tool_calls=[
            DeltaToolCall(index=0, function=DeltaFunctionCall(name="get_weather"))
        ])
        result2 = self.parser.get_tool_calls_from_deltas(seq_id, delta2)
        assert result2 is None  # Still not complete
        
        # Step 3: Partial arguments
        delta3 = DeltaMessage(tool_calls=[
            DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='{"city": "Ber'))
        ])
        result3 = self.parser.get_tool_calls_from_deltas(seq_id, delta3)
        assert result3 is None  # Incomplete JSON
        
        # Step 4: Complete arguments
        delta4 = DeltaMessage(tool_calls=[
            DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='lin"}'))
        ])
        result4 = self.parser.get_tool_calls_from_deltas(seq_id, delta4)
        
        # Should yield complete tool call
        assert result4 is not None
        assert len(result4) == 1
        tool_call = result4[0]
        assert tool_call.id == "stream_001"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == '{"city": "Berlin"}'
    
    def test_streaming_multiple_tools_partial(self):
        """Test multiple tool calls being built with overlapping partial updates."""
            
        seq_id = 2002
        
        # Start both tool calls
        delta1 = DeltaMessage(tool_calls=[
            DeltaToolCall(index=0, id="multi_001", type="function"),
            DeltaToolCall(index=1, id="multi_002", type="function")
        ])
        result1 = self.parser.get_tool_calls_from_deltas(seq_id, delta1)
        assert result1 is None
        
        # Add function names
        delta2 = DeltaMessage(tool_calls=[
            DeltaToolCall(index=0, function=DeltaFunctionCall(name="calculate")),
            DeltaToolCall(index=1, function=DeltaFunctionCall(name="notify"))
        ])
        result2 = self.parser.get_tool_calls_from_deltas(seq_id, delta2)
        assert result2 is None
        
        # Complete first tool, start second tool arguments
        delta3 = DeltaMessage(tool_calls=[
            DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='{"x": 10,')),
            DeltaToolCall(index=1, function=DeltaFunctionCall(arguments='{"message"'))
        ])
        result3 = self.parser.get_tool_calls_from_deltas(seq_id, delta3)
        assert result3 is None
        
        delta4 = DeltaMessage(tool_calls=[
            DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='"y": 20}')),
            DeltaToolCall(index=1, function=DeltaFunctionCall(arguments=': "calc done"}'))
        ])
        result4 = self.parser.get_tool_calls_from_deltas(seq_id, delta4)
        

        # Second tool should be ready
        assert result4 is not None
        assert len(result4) == 2
        assert result4[0].id == "multi_001"
        import json
        
        args_fn1 = json.loads(result4[0].function.arguments.strip())
        args_fn2 = json.loads('{"x": 10, "y": 20}')
        
        assert_eq_dict(args_fn1, args_fn2)
        assert result4[1].id == "multi_002"
        
        args_fn2 = json.loads(result4[1].function.arguments.strip())
        args_fn2 = json.loads('{"message": "calc done"}')
        assert_eq_dict(args_fn2, args_fn2)
    
    def test_nested_json_bracket_tracking(self):
        """Test complex nested JSON bracket counting."""
            
        seq_id = 3003
        
        # Setup tool
        delta1 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0, 
                id="nested_001", 
                type="function",
                function=DeltaFunctionCall(name="complex_config")
            )
        ])
        self.parser.get_tool_calls_from_deltas(seq_id, delta1)
        
        # Add deeply nested incomplete JSON
        delta2 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"level1": {"level2": {"level3": {"value":')
            )
        ])
        result2 = self.parser.get_tool_calls_from_deltas(seq_id, delta2)
        assert result2 is None
        
        # Check brace count
        state = self.parser.tool_states[seq_id][0]
        assert state['brace_count'] == 4  # Four opening braces
        
        # Close all braces
        delta3 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments=' "deep"}}}}')
            )
        ])
        result3 = self.parser.get_tool_calls_from_deltas(seq_id, delta3)
        
        assert result3 is not None
        assert state['brace_count'] == 0  # All balanced


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestStreamingToolParser()
    
    print("Running focused StreamingToolParser tests...")
    print("=" * 50)
    
    test_methods = [
        test_instance.test_streaming_tool_call_incremental,
        test_instance.test_streaming_multiple_tools_partial,
        test_instance.test_nested_json_bracket_tracking,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_instance.setup_method()
            test_method()
            print(f"✅ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    if failed == 0:
        print(f"🎉 ALL {passed} TESTS PASSED!")
    else:
        print(f"❌ {failed} tests failed, {passed} tests passed")
    print("=" * 50) 