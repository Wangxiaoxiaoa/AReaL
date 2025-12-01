import asyncio
import base64
import io
import os
import uuid
from typing import Any

import requests
import torch
from PIL import Image
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from areal.utils import logging

logger = logging.getLogger("RLENV workflow")


class RLENVMultiModelWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor,
        max_turns: int = 6,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        controller_base: str | None = None,
        auth_token_env: str = "ENV_WORKER_TOKEN",
        keep_session_open: bool = True,
        n_trajs: int | None = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_turns = max_turns
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.controller_base = (
            controller_base
            or os.environ.get("CONTROLLER_BASE_URL")
            or "http://127.0.0.1:8081"
        )
        self.auth_token_env = auth_token_env
        self.keep_session_open = keep_session_open
        self._bound_session_id: str | None = None
        self._bound_worker_base: str | None = None
        self.n_trajs = n_trajs if n_trajs is not None else self.gconfig.n_samples

    def bind_session(self, alloc: dict[str, Any]):
        self._bound_session_id = alloc.get("session_id")
        host = alloc.get("worker_host")
        port = alloc.get("worker_port")
        if host and port:
            self._bound_worker_base = f"http://{host}:{int(port)}"
        else:
            self._bound_worker_base = None

    def _decode_image(self, b64: str) -> Image.Image:
        b = base64.b64decode(b64)
        return Image.open(io.BytesIO(b)).convert("RGB")

    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        headers = {}
        token = os.environ.get(self.auth_token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async def _run_one():
            local_session = False
            session_id = None
            worker_base = None
            payload: dict[str, Any] = {"task_config": data.get("config")}
            start_resp = await asyncio.to_thread(
                requests.post,
                f"{self.controller_base}/session/start",
                headers=headers,
                json=payload,
                timeout=60,
            )
            start_resp.raise_for_status()
            alloc = start_resp.json()
            session_id = alloc["session_id"]
            worker_base = f"http://{alloc['worker_host']}:{int(alloc['worker_port'])}"
            local_session = True

            obs_resp = await asyncio.to_thread(
                requests.get, f"{worker_base}/observe", headers=headers, timeout=30
            )
            obs_resp.raise_for_status()
            observation = obs_resp.json() or {}
            screenshot_b64 = observation.get("screenshot_b64") or ""

            seq = []
            logprobs = []
            loss_mask = []
            versions = []
            rewards = 0.0
            turns = 0
            pixel_values = None

            try:
                while turns < self.max_turns and isinstance(screenshot_b64, str) and len(screenshot_b64) > 0:
                    turns += 1
                    img = self._decode_image(screenshot_b64)
                    sys_prompt = data.get("system")
                    user_prompt = data.get("instruct")
                    messages = []
                    if isinstance(sys_prompt, str) and len(sys_prompt) > 0:
                        messages.append({"role": "system", "content": sys_prompt})
                    if isinstance(user_prompt, str) and len(user_prompt) > 0:
                        messages.append({"role": "user", "content": user_prompt})
                    if not messages:
                        messages = [
                            {
                                "role": "user",
                                "content": "You are controlling a desktop via Python. Generate pyautogui code in triple backticks to progress the task.",
                            }
                        ]

                    processed = self.processor(
                        images=[img], text=messages, padding=False, return_tensors="pt"
                    )
                    input_ids = processed["input_ids"].tolist()[0]
                    pixel_values = processed.get("pixel_values")

                    req = ModelRequest(
                        rid=uuid.uuid4().hex,
                        input_ids=input_ids,
                        image_data=[screenshot_b64],
                        gconfig=self.gconfig.new(n_samples=1),
                        tokenizer=self.tokenizer,
                        processor=self.processor,
                    )

                    resp = await engine.agenerate(req)

                    seq.extend(resp.input_tokens + resp.output_tokens)
                    logprobs.extend([0.0] * resp.input_len + resp.output_logprobs)
                    loss_mask.extend([0] * resp.input_len + [1] * resp.output_len)
                    versions.extend([-1] * resp.input_len + resp.output_versions)

                    completion_str = self.tokenizer.decode(resp.output_tokens)
                    step_resp = await asyncio.to_thread(
                        requests.post,
                        f"{self.controller_base}/session/step",
                        headers=headers,
                        json={
                            "session_id": session_id,
                            "output": completion_str,
                            "extras": {"pause": 1},
                            "pause": 1,
                        },
                        timeout=120,
                    )
                    if step_resp.status_code != 200:
                        break
                    step_data = step_resp.json() or {}
                    observation = step_data.get("observation") or {}
                    screenshot_b64 = observation.get("screenshot_b64") or ""
                    results = step_data.get("results") or []
                    if results:
                        info = results[-1].get("info") or {}
                        done = bool(info.get("done"))
                        fail = bool(info.get("fail"))
                        if done or fail:
                            rewards = 1.0 if done else 0.0
                            break
            except Exception:
                pass
            finally:
                try:
                    if session_id is not None and (local_session or not self.keep_session_open):
                        await asyncio.to_thread(
                            requests.post,
                            f"{self.controller_base}/session/finish",
                            headers=headers,
                            json={"session_id": session_id, "success": bool(rewards > 0)},
                            timeout=30,
                        )
                except Exception:
                    pass

            seq_t = torch.tensor(seq, dtype=torch.int32).unsqueeze(0)
            logp_t = torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0)
            lmask_t = torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0)
            ver_t = torch.tensor(versions, dtype=torch.int32).unsqueeze(0)
            attn_t = torch.ones(len(seq), dtype=torch.bool).unsqueeze(0)
            rew_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
            res = {
                "input_ids": seq_t,
                "logprobs": logp_t,
                "loss_mask": lmask_t,
                "versions": ver_t,
                "attention_mask": attn_t,
                "rewards": rew_t,
            }
            if pixel_values is not None:
                res["multi_modal_input"] = [{"pixel_values": pixel_values}]
            return res

        tasks = [asyncio.create_task(_run_one()) for _ in range(max(1, int(self.n_trajs)))]
        results = []
        for t in asyncio.as_completed(tasks):
            r = await t
            results.append(r)
        return concat_padded_tensors(results)
