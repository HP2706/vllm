# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List

from vllm.config import SchedulerConfig, SpecialKwargs
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.output_processor.streaming_tool_parser import StreamingToolParser
from vllm.logger import init_logger
from vllm.sequence import (CompletionSequenceGroupOutput, SequenceGroup, 
                          SequenceGroupOutput, SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

logger = init_logger(__name__)

class SequenceHookOutputProcessor(SingleStepOutputProcessor):
    """Enhanced SingleStepOutputProcessor that handles special tokens and 
    sequence lifecycle management (sync/promise tokens, dependency handling).
    
    This processor extends the standard single-step output processing with
    custom sequence hook logic for:
    1. Handling <sync> tokens to pause/resume sequences
    2. Handling <promise> tokens to spawn new dependent sequences  
    3. Managing parent-child sequence dependencies
    4. Triggering prefill when child sequences finish
    """

    def __init__(self, scheduler_config: SchedulerConfig,
                 detokenizer: Detokenizer, scheduler: List[Scheduler],
                 seq_counter: Counter, stop_checker: StopChecker,
                 streaming_tool_parser: StreamingToolParser,
                 special_kwargs: SpecialKwargs):
        super().__init__(
            scheduler_config=scheduler_config,
            detokenizer=detokenizer,
            scheduler=scheduler,
            seq_counter=seq_counter,
            stop_checker=stop_checker,
            streaming_tool_parser=streaming_tool_parser
        )
        self.special_kwargs = special_kwargs
        
    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput],
                        is_async: bool) -> None:
        """Enhanced process_outputs that includes sequence hook logic."""
        
        # First run our custom sequence hook logic
        self._run_sequence_hook(sequence_group, outputs)
        
        # Then run the standard output processing
        super().process_outputs(sequence_group, outputs, is_async)
        
    def _run_sequence_hook(self, seq_group: SequenceGroup, 
                          outputs: List[SequenceGroupOutput]) -> None:
        """Run the sequence hook to handle special tokens and manage sequence lifecycle."""
        if not outputs or not isinstance(outputs[0], CompletionSequenceGroupOutput):
            return
            
        output = outputs[0]  # Single step only has one output
        
        sync_token_id = self.special_kwargs.sync_token_id
        promise_token_id = self.special_kwargs.promise_token_id
        
        # Handle sync tokens
        self._handle_sync(seq_group, output, sync_token_id)
        
        # Handle promise tokens  
        self._handle_promise(seq_group, output, promise_token_id)
        
        # Handle unblocking of parent sequences when children finish
        self._handle_unblock(seq_group)
        
    def _handle_sync(self, seq_group: SequenceGroup, 
                    output: CompletionSequenceGroupOutput, 
                    sync_token_id: int) -> None:
        """Handle <sync> tokens to pause current sequence and resume new sequence."""
        for seq_idx, sample in enumerate(output.samples):
            if sample.output_token == sync_token_id:
                logger.info(f"Sync token detected - pausing sequence {seq_idx}")
                # Find the sequence by index and block it
                if seq_idx < len(seq_group.seqs):
                    seq = seq_group.seqs[seq_idx]
                    seq.status = SequenceStatus.BLOCKED
                    logger.info(f"Blocked sequence {seq.seq_id}")
                    
    def _handle_promise(self, seq_group: SequenceGroup,
                       output: CompletionSequenceGroupOutput,
                       promise_token_id: int) -> None:
        """Handle <promise> tokens to spawn new dependent sequences."""
        for seq_idx, sample in enumerate(output.samples):
            if sample.output_token == promise_token_id:
                logger.info(f"Promise token detected - spawning child sequence for {seq_idx}")
                
                # Find parent sequence
                if seq_idx < len(seq_group.seqs):
                    parent_seq = seq_group.seqs[seq_idx]
                    
                    # Generate new sequence ID
                    child_seq_id = next(self.seq_counter)
                    
                    # Spawn child sequence with additional prompt
                    child_seq = seq_group.spawn_child(
                        parent_seq_id=parent_seq.seq_id,
                        new_seq_id=child_seq_id,
                        # Add any additional spawn parameters here
                    )
                    
                    # Block parent until child finishes
                    parent_seq.status = SequenceStatus.BLOCKED
                    
                    logger.info(f"Spawned child sequence {child_seq_id} from parent {parent_seq.seq_id}")

    def _handle_single_child_finished(self, seq_group: SequenceGroup) -> None:
        '''
        When a child is finished, we need to update the parent with the result, this triggers a prefill
        We do this so we dont have to wait for all children to finish before we can resume the parent
        NOTE we assume ordering of children outputs does not matter
        '''
        #TODO: Implement this
        pass

    def _handle_unblock(self, seq_group: SequenceGroup) -> None:
        """Handle unblocking of parent sequences when all child sequences finish."""
        for seq in seq_group.seqs:
            if seq.is_blocked and seq.is_parent:
                # Check if all children are finished
                all_children_finished = all(
                    child_seq.is_finished() 
                    for child_seq in seq_group.seqs 
                    if child_seq.parent_seq_id == seq.seq_id
                )
                
                if all_children_finished:
                    # Unblock the parent sequence
                    seq.status = SequenceStatus.WAITING
                    logger.info(f"Unblocked parent sequence {seq.seq_id} - all children finished")
                    
                    # Trigger prefill by updating with child results
                    self._merge_child_results(seq, seq_group)
                    
    def _merge_child_results(self, parent_seq, seq_group: SequenceGroup) -> None:
        """Merge results from child sequences back into parent and trigger prefill."""
        # Find all finished children
        child_seqs = [
            seq for seq in seq_group.seqs 
            if seq.parent_seq_id == parent_seq.seq_id and seq.is_finished()
        ]
        
        # Merge child outputs into parent (implement your specific merge logic)
        merged_output = self._compute_merged_output(child_seqs)
        
        # Append merged content to parent sequence for prefill
        if merged_output:
            # This would trigger a prefill on the next scheduling cycle
            parent_seq.data.append_prompt_tokens(merged_output)
            logger.info(f"Merged {len(child_seqs)} child results into parent {parent_seq.seq_id}")
            
    def _compute_merged_output(self, child_seqs) -> str:
        """Compute the merged output from child sequences."""
        # Implement your specific logic for merging child sequence outputs
        # This is a placeholder - replace with your actual merge strategy
        merged_outputs = []
        for child_seq in child_seqs:
            if child_seq.output_text:
                merged_outputs.append(child_seq.output_text)
        
        return "\n".join(merged_outputs) if merged_outputs else "" 