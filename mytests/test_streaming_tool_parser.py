#!/usr/bin/env python3
"""
Comprehensive tests for StreamingToolParser with real DeepSeekV3 tokenizer.
Tests V1 architecture integration and streaming tool call parsing functionality.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer
TRANSFORMERS_AVAILABLE = True


from vllm.entrypoints.openai.protocol import (
    DeltaMessage, DeltaToolCall, DeltaFunctionCall
)
from vllm.v1.engine.streaming_tool_parser import StreamingToolParser
from vllm.entrypoints.openai.tool_parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser
from vllm.v1.engine import EngineCoreOutput

class TestStreamingToolParserV1:
    """Test class for V1 StreamingToolParser with real DeepSeek tokenizer."""
    
    @classmethod
    def setup_class(cls):
       
        cls.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v2.5", trust_remote_code=True)
        cls.tool_parser = DeepSeekV3ToolParser(cls.tokenizer)
        cls.streaming_parser = StreamingToolParser(cls.tokenizer, cls.tool_parser)
        
    def test_incomplete_tool_call_handling_and_completion(self):
        """Test that incomplete tool call deltas are handled properly and complete tool calls are returned correctly."""
        request_id = "incremental_tool_request"
        
        # Define the expected complete tool call that we're building towards
        expected_id = "call_weather_123"
        expected_type = "function"
        expected_name = "get_weather"
        expected_args = '{"city": "San Francisco", "units": "celsius"}'
        
        # Step 1: Send incomplete delta with just ID and type
        delta1 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                id="call_weather_123",
                type="function"
            )
        ])
        
        result1 = self.streaming_parser.get_tool_calls_from_deltas(request_id, delta1)
        assert result1 is None, "Should return None for incomplete tool call (only ID and type)"
        
        # Step 2: Add function name - still incomplete
        delta2 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(name="get_weather")
            )
        ])
        
        result2 = self.streaming_parser.get_tool_calls_from_deltas(request_id, delta2)
        assert result2 is None, "Should return None for incomplete tool call (missing arguments)"
        
        # Step 3: Add partial arguments - still incomplete
        delta3 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"city": "San Francisco"')
            )
        ])
        
        result3 = self.streaming_parser.get_tool_calls_from_deltas(request_id, delta3)
        assert result3 is None, "Should return None for incomplete tool call (invalid JSON arguments)"
        
        # Step 4: Complete the arguments - now should return complete tool call
        delta4 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments=', "units": "celsius"}')
            )
        ])
        
        result4 = self.streaming_parser.get_tool_calls_from_deltas(request_id, delta4)
        assert result4 is not None, "Should return complete tool call when all parts are provided"
        assert len(result4) == 1, "Should return exactly one tool call"
        
        # Verify the returned tool call matches our expected tool call
        returned_tool_call = result4[0]
        
        assert returned_tool_call.id == expected_id, f"Tool call ID mismatch: {returned_tool_call.id} != {expected_id}"
        assert returned_tool_call.type == expected_type, f"Tool call type mismatch: {returned_tool_call.type} != {expected_type}"
        assert returned_tool_call.function.name == expected_name, f"Function name mismatch: {returned_tool_call.function.name} != {expected_name}"
        assert returned_tool_call.function.arguments == expected_args, f"Function arguments mismatch: {returned_tool_call.function.arguments} != {expected_args}"
        
        # Verify state tracking
        assert request_id in self.streaming_parser.tool_states, "Request should be tracked in tool_states"
        assert 0 in self.streaming_parser.tool_states[request_id], "Tool call index 0 should be tracked"
        assert request_id in self.streaming_parser.yielded_indices, "Request should be tracked in yielded_indices"
        assert 0 in self.streaming_parser.yielded_indices[request_id], "Tool call index 0 should be marked as yielded"
        
        print(f"✓ Successfully tested incremental tool call building for request {request_id}")
        print(f"✓ Verified returned tool call matches expected: id={expected_id}, name={expected_name}, args={expected_args}")
        
    def test_multiple_tool_calls(self):
        """Test handling of multiple tool calls in sequence."""
        request_id = "multi_tool_request"
        
        # Two complete tool calls
        delta = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                id="call_1",
                type="function",
                function=DeltaFunctionCall(
                    name="get_weather",
                    arguments='{"city": "NYC"}'
                )
            ),
            DeltaToolCall(
                index=1,
                id="call_2",
                type="function",
                function=DeltaFunctionCall(
                    name="send_email",
                    arguments='{"to": "test@example.com"}'
                )
            )
        ])
        
        # Process should return ready tool calls
        result = self.streaming_parser.get_tool_calls_from_deltas(request_id, delta)
        
        # Both tool calls should be returned since they're complete
        assert result is not None
        assert len(result) == 2
        assert result[0].id == "call_1"
        assert result[1].id == "call_2"

    def test_streaming_tool_parser_v1(self):
        
        
        text = '''<｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"city": "New York", "units": "celsius"}
    ```<｜tool▁call▁end｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>send_email
    ```json
    {"to": "john@example.com", "subject": "Weather Update", "body": "The weather in New York is sunny."}
    ```<｜tool▁call▁end｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>calculate_distance
    ```json
    {"origin": "New York", "destination": "Los Angeles", "mode": "driving"}
    ```<｜tool▁call▁end｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>create_calendar_event
    ```json
    {"title": "Team Meeting", "date": "2024-01-15", "time": "14:00", "duration": 60, "attendees": ["alice@company.com", "bob@company.com"]}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>'''
        
        tokens: list[int] = self.tokenizer.encode(text, add_special_tokens=False)
        # create token chunks

        
        tool_calls = []
        for i in range(0, len(tokens), 3):
            chunk = tokens[i:i+3]
            engine_output = EngineCoreOutput(
                request_id="test",
                new_token_ids=chunk
            )
            tool_calls = self.streaming_parser.process_engine_output_for_tools(engine_output)
            if tool_calls:
                tool_calls.append(tool_calls[0])

        assert len(tool_calls) == 4, f"Expected 4 tool calls, got {len(tool_calls)}"
        assert tool_calls[0].function.name == "get_weather", f"Expected get_weather, got {tool_calls[0].function.name}"
        assert tool_calls[1].function.name == "send_email", f"Expected send_email, got {tool_calls[1].function.name}"
        assert tool_calls[2].function.name == "calculate_distance", f"Expected calculate_distance, got {tool_calls[2].function.name}"
        assert tool_calls[3].function.name == "create_calendar_event", f"Expected create_calendar_event, got {tool_calls[3].function.name}"


        
    def test_incomplete_json_handling(self):
        """Test handling of incomplete JSON in tool call arguments."""
        request_id = "incomplete_json_request"
        
        # Set up initial tool call
        delta1 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                id="call_json_test",
                type="function",
                function=DeltaFunctionCall(name="test_func")
            )
        ])
        self.streaming_parser.get_tool_calls_from_deltas(request_id, delta1)
        
        # Add incomplete JSON
        delta2 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"key": "val')
            )
        ])
        
        result2 = self.streaming_parser.get_tool_calls_from_deltas(request_id, delta2)
        assert result2 is None  # Incomplete JSON, should not yield
        
        # Complete the JSON
        delta3 = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='ue"}')
            )
        ])
        
        result3 = self.streaming_parser.get_tool_calls_from_deltas(request_id, delta3)
        assert result3 is not None  # Now complete
        assert result3[0].function.arguments == '{"key": "value"}'
    
        
    def test_error_handling(self):
        """Test error handling in tool parsing."""
        request_id = "error_handling_test"
        
        # Test with empty token list
        empty_output = EngineCoreOutput(
            request_id=request_id,
            new_token_ids=[]
        )
        
        delta_message = self.streaming_parser.update_from_engine_output(empty_output)
        assert delta_message is None
        
        # Test with malformed delta message
        malformed_delta = DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=0,
                id=None,  # Missing required field
                type="function"
            )
        ])
        
        # Should handle gracefully
        result = self.streaming_parser.get_tool_calls_from_deltas(request_id, malformed_delta)
        assert result is None, f"Should return None for malformed delta: {result}"
        # Should not crash, might return None

def run_tests():
    """Run all tests with proper error handling."""
    test_class = TestStreamingToolParserV1()
    

    test_class.setup_class()

        
    tests = [
        test_class.test_incomplete_tool_call_handling_and_completion,
        test_class.test_multiple_tool_calls,
        test_class.test_incomplete_json_handling,
        test_class.test_error_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n🏁 Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Starting V1 StreamingToolParser Tests with Real DeepSeek Tokenizer...")
    print("=" * 80)
    
    success = run_tests()
    
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("💥 SOME TESTS FAILED!")
        sys.exit(1) 