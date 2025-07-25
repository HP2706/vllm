from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm.entrypoints.openai.tool_parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser
from vllm.v1.engine.streaming_tool_parser import StreamingToolParser, EngineCoreOutput


if __name__ == "__main__":
    tokenizer : PreTrainedTokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v2.5", trust_remote_code=True)
    tool_parser = DeepSeekV3ToolParser(tokenizer)
    streaming_parser = StreamingToolParser(tokenizer, tool_parser)
    