# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch


EXTRA_ARGS_KEY = "cross_batch_attention"


@dataclass(frozen=True)
class CrossBatchAttentionParams:
    """Per-request cross-batch attention configuration."""

    group_id: str
    replica_id: int
    group_size: int
    virtual_token_id: int
    virtual_window_size: int

    @classmethod
    def from_extra_args(
        cls, extra_args: dict[str, Any] | None
    ) -> "CrossBatchAttentionParams | None":
        if not extra_args or EXTRA_ARGS_KEY not in extra_args:
            return None

        raw = extra_args[EXTRA_ARGS_KEY]
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"{EXTRA_ARGS_KEY} must be a dict when provided")
        if not raw.get("enabled", True):
            return None

        missing = [
            key
            for key in (
                "group_id",
                "replica_id",
                "group_size",
                "virtual_token_id",
                "virtual_window_size",
            )
            if key not in raw
        ]
        if missing:
            raise ValueError(
                f"{EXTRA_ARGS_KEY} is missing required field(s): "
                f"{', '.join(missing)}"
            )

        group_id = raw["group_id"]
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"{EXTRA_ARGS_KEY}.group_id must be a non-empty string")

        replica_id = int(raw["replica_id"])
        group_size = int(raw["group_size"])
        virtual_token_id = int(raw["virtual_token_id"])
        virtual_window_size = int(raw["virtual_window_size"])
        if group_size <= 0:
            raise ValueError(f"{EXTRA_ARGS_KEY}.group_size must be positive")
        if not 0 <= replica_id < group_size:
            raise ValueError(
                f"{EXTRA_ARGS_KEY}.replica_id must be in [0, group_size)"
            )
        if virtual_window_size <= 0:
            raise ValueError(
                f"{EXTRA_ARGS_KEY}.virtual_window_size must be positive"
            )

        return cls(
            group_id=group_id,
            replica_id=replica_id,
            group_size=group_size,
            virtual_token_id=virtual_token_id,
            virtual_window_size=virtual_window_size,
        )


@dataclass(frozen=True)
class CrossBatchAttentionData:
    """Scheduler-level cross-batch data keyed by request id."""

    params_by_req_id: dict[str, CrossBatchAttentionParams]

    @classmethod
    def from_scheduled_requests(
        cls, requests: list[Any]
    ) -> "CrossBatchAttentionData | None":
        params_by_req_id = {
            req.request_id: req.cross_batch_attention
            for req in requests
            if getattr(req, "cross_batch_attention", None) is not None
        }
        if not params_by_req_id:
            return None
        return cls(params_by_req_id=params_by_req_id)


@dataclass
class CrossBatchAttentionMetadata:
    """Worker-order cross-batch tensors for attention backends."""

    enabled: torch.Tensor
    group_ids: torch.Tensor
    replica_ids: torch.Tensor
    group_members: torch.Tensor
    group_member_mask: torch.Tensor
    virtual_token_ids: torch.Tensor
    virtual_window_sizes: torch.Tensor
    token_ids: torch.Tensor
