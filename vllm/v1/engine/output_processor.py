# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import torch

from vllm.outputs import (CompletionOutput, PoolingOutput,
                          PoolingRequestOutput, RequestOutput)
from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import (IterationStats, LoRARequestStates,
                                   RequestStateStats)


class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    def __init__(self, output_kind: RequestOutputKind):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.output: Optional[Union[RequestOutput, PoolingRequestOutput,
                                    Exception]] = None
        self.ready = asyncio.Event()

    def put(self, output: Union[RequestOutput, PoolingRequestOutput,
                                Exception]) -> None:
        """Non-blocking put operation."""
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, (RequestOutput, PoolingRequestOutput)):
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            self.output.add(output, aggregate=self.aggregate)

    async def get(self) -> Union[RequestOutput, PoolingRequestOutput]:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def get_nowait(
            self) -> Optional[Union[RequestOutput, PoolingRequestOutput]]:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output


@dataclass
class OutputProcessorOutput:

    request_outputs: list[Union[RequestOutput, PoolingRequestOutput]]
    reqs_to_abort: list[str]


class RequestState:

    def __init__(
        self,
        request_id: str,
        parent_req: Optional[ParentRequest],
        request_index: int,
        lora_name: Optional[str],
        output_kind: RequestOutputKind,
        prompt: Optional[str],
        prompt_token_ids: list[int],
        logprobs_processor: Optional[LogprobsProcessor],
        detokenizer: Optional[IncrementalDetokenizer],
        max_tokens_param: Optional[int],
        arrival_time: float,
        queue: Optional[RequestOutputCollector],
        log_stats: bool,
    ):
        self.request_id = request_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.lora_name = lora_name
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_len = len(prompt_token_ids)
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.max_tokens_param = max_tokens_param
        self.is_prefilling = True
        self.queue = queue

        self.stats = RequestStateStats(
            arrival_time=arrival_time) if log_stats else None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: EngineCoreRequest,
        prompt: Optional[str],
        parent_req: Optional[ParentRequest],
        request_index: int,
        queue: Optional[RequestOutputCollector],
        log_stats: bool,
    ) -> "RequestState":

        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        return cls(
            request_id=request.request_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_name=(request.lora_request.name
                       if request.lora_request is not None else None),
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
        )

    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: Optional[torch.Tensor],
        finish_reason: Optional[FinishReason],
        stop_reason: Union[int, str, None],
        kv_transfer_params: Optional[dict[str, Any]] = None,
        num_cached_tokens: int = 0,
    ) -> Optional[Union[RequestOutput, PoolingRequestOutput]]:

        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        request_id = self.request_id
        if pooling_output is not None:
            return self._new_request_output(
                request_id, [self._new_pooling_output(pooling_output)],
                finished)

        output = self._new_completion_output(new_token_ids, finish_reason,
                                             stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            request_id, outputs, finished = self.parent_req.get_outputs(
                request_id, output)
            if not outputs:
                return None

        return self._new_request_output(request_id, outputs, finished,
                                        kv_transfer_params, num_cached_tokens)

    def _new_request_output(
        self,
        request_id: str,
        outputs: Union[list[CompletionOutput], list[PoolingOutput]],
        finished: bool,
        kv_transfer_params: Optional[dict[str, Any]] = None,
        num_cached_tokens: int = 0,
    ) -> Union[RequestOutput, PoolingRequestOutput]:

        if isinstance(outputs[0], PoolingOutput):
            assert len(outputs) == 1
            return PoolingRequestOutput(
                request_id=request_id,
                outputs=outputs[0],
                prompt_token_ids=self.prompt_token_ids,
                finished=finished,
            )
        assert self.logprobs_processor is not None
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=request_id,
            prompt=self.prompt,
            prompt_token_ids=self.prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            num_cached_tokens=num_cached_tokens,
        )

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: Optional[FinishReason],
        stop_reason: Union[int, str, None],
    ) -> CompletionOutput:

        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            logprobs = logprobs[-len(token_ids):]

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None)

    def _new_pooling_output(
        self,
        pooling_output: torch.Tensor,
    ) -> PoolingOutput:

        return PoolingOutput(data=pooling_output)


class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    def __init__(
        self,
        tokenizer: TokenizerGroup,
        log_stats: bool,
    ):
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.lora_states = LoRARequestStates()

    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""

        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    def abort_requests(
        self,
        request_ids: Iterable[str],
    ) -> list[str]:
        request_ids_to_abort = []
        for request_id in request_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                self.lora_states.abort_request(req_state)
                request_ids_to_abort.append(request_id)
            else:
                parent = self.parent_requests.pop(request_id, None)
                if parent and parent.child_requests:
                    self.abort_requests(parent.child_requests)
                    request_ids_to_abort.extend(parent.child_requests)
        return request_ids_to_abort

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: Optional[str],
        parent_req: Optional[ParentRequest] = None,
        request_index: int = 0,
        queue: Optional[RequestOutputCollector] = None,
    ) -> None:
        request_id = request.request_id
        if request_id in self.request_states:
            raise ValueError(f"Request id {request_id} already running.")

        tokenizer = None if not self.tokenizer else \
            self.tokenizer.get_lora_tokenizer(request.lora_request)

        req_state = RequestState.from_new_request(tokenizer=tokenizer,
                                                  request=request,
                                                  prompt=prompt,
                                                  parent_req=parent_req,
                                                  request_index=request_index,
                                                  queue=queue,
                                                  log_stats=self.log_stats)
        self.request_states[request_id] = req_state
        self.lora_states.add_request(req_state)
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: Optional[float] = None,
        iteration_stats: Optional[IterationStats] = None,
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM), 
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine), 
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the 
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """

        request_outputs: Union[list[RequestOutput],
                               list[PoolingRequestOutput]] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(req_state, engine_core_output,
                                           engine_core_timestamp,
                                           iteration_stats)

            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            num_cached_tokens = engine_core_output.num_cached_tokens
            req_state.is_prefilling = False

            if pooling_output is None:
                assert req_state.detokenizer is not None
                assert req_state.logprobs_processor is not None
                # 2) Detokenize the token ids into text and perform stop checks.
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP)
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(
                    engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                    new_token_ids, pooling_output, finish_reason, stop_reason,
                    kv_transfer_params, num_cached_tokens):
                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                self.request_states.pop(req_id)
                # Remove parent request if applicable.
                parent_req = req_state.parent_req
                if parent_req and not parent_req.child_requests:
                    self.parent_requests.pop(parent_req.request_id, None)
                if not engine_core_output.finished:
                    # If req not finished in EngineCore, but Detokenizer
                    # detected stop string, abort needed in EngineCore.
                    reqs_to_abort.append(req_id)

                # Track per-request stats
                self._update_stats_from_finished(req_state, finish_reason,
                                                 iteration_stats)

        self.lora_states.update_iteration_stats(iteration_stats)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    def _update_stats_from_output(self, req_state: RequestState,
                                  engine_core_output: EngineCoreOutput,
                                  engine_core_timestamp: Optional[float],
                                  iteration_stats: Optional[IterationStats]):
        if iteration_stats is None:
            return

        lora_stats = self.lora_states.get_stats(req_state)

        assert engine_core_timestamp is not None
        assert req_state.stats is not None
        iteration_stats.update_from_output(engine_core_output,
                                           engine_core_timestamp,
                                           req_state.is_prefilling,
                                           req_state.prompt_len,
                                           req_state.stats, lora_stats)

    def _update_stats_from_finished(self, req_state: RequestState,
                                    finish_reason: Optional[FinishReason],
                                    iteration_stats: Optional[IterationStats]):
        if iteration_stats is None:
            return

        assert finish_reason is not None
        assert req_state.stats is not None
        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            num_prompt_tokens=len(req_state.prompt_token_ids),
            max_tokens_param=req_state.max_tokens_param,
            req_stats=req_state.stats)
        self.lora_states.finish_request(req_state)

        ParentRequest.observe_finished_request(
            req_state.parent_req, iteration_stats,
            req_state.stats.num_generation_tokens)
