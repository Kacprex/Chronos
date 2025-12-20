from __future__ import annotations

import time
import uuid
import queue
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class InferenceRequest:
    worker_id: int
    req_id: str
    batch: np.ndarray  # shape (B, C, 8, 8), float32


class BatchedInferenceServer(mp.Process):
    """
    A single-process inference server that owns the GPU model.
    Many worker processes can send inference requests; the server batches them.

    Request item format:
      (worker_id: int, req_id: str, batch: np.ndarray[B,C,8,8])

    Response item format (per-worker response queue):
      (req_id: str, policy_logits: np.ndarray[B,MOVE_SPACE], values: np.ndarray[B,1])
    """
    def __init__(
        self,
        model_path: str,
        device: str,
        request_q: mp.Queue,
        response_qs: List[mp.Queue],
        max_batch: int = 64,
        max_wait_ms: int = 2,
        use_amp: bool = True,
    ):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.device = device
        self.request_q = request_q
        self.response_qs = response_qs
        self.max_batch = int(max_batch)
        self.max_wait_ms = int(max_wait_ms)
        self.use_amp = bool(use_amp)

    def run(self) -> None:
        torch.set_num_threads(1)

        device = torch.device(self.device)
        model = torch.load(self.model_path, map_location=device)
        model.eval()

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        stop = False
        while not stop:
            reqs: List[InferenceRequest] = []
            total = 0

            t0 = time.perf_counter()
            wait_s = max(0.0001, self.max_wait_ms / 1000.0)

            # Gather requests for up to max_wait_ms (or until max_batch reached)
            while total < self.max_batch:
                timeout = max(0.0, wait_s - (time.perf_counter() - t0))
                if timeout <= 0 and reqs:
                    break
                try:
                    item = self.request_q.get(timeout=timeout if timeout > 0 else 0.0)
                except queue.Empty:
                    break

                # Stop signal
                if item is None:
                    stop = True
                    break

                worker_id, req_id, batch = item
                if batch is None:
                    stop = True
                    break

                if not isinstance(batch, np.ndarray):
                    batch = np.asarray(batch, dtype=np.float32)
                if batch.dtype != np.float32:
                    batch = batch.astype(np.float32, copy=False)

                reqs.append(InferenceRequest(worker_id=int(worker_id), req_id=str(req_id), batch=batch))
                total += int(batch.shape[0])

            if not reqs:
                continue

            big_batch = np.concatenate([r.batch for r in reqs], axis=0)  # (N, C, 8, 8)
            x = torch.from_numpy(big_batch).to(device, non_blocking=(device.type == "cuda"))

            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                        policy_logits, values = model(x)
                else:
                    policy_logits, values = model(x)

            policy_logits = policy_logits.float().cpu().numpy()
            values = values.float().cpu().numpy()

            offset = 0
            for r in reqs:
                b = int(r.batch.shape[0])
                pl = policy_logits[offset:offset + b]
                v = values[offset:offset + b]
                offset += b
                self.response_qs[r.worker_id].put((r.req_id, pl, v))


class BatchedInferenceClient:
    """
    Used inside self-play worker processes to talk to BatchedInferenceServer.
    """
    def __init__(self, worker_id: int, request_q: mp.Queue, response_q: mp.Queue):
        self.worker_id = int(worker_id)
        self.request_q = request_q
        self.response_q = response_q

    def infer_batch(self, batch: np.ndarray, timeout_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        req_id = uuid.uuid4().hex
        self.request_q.put((self.worker_id, req_id, batch))
        while True:
            rid, pl, v = self.response_q.get(timeout=timeout_s)
            if rid == req_id:
                return pl, v
