import os
import subprocess
import socket
import time
import sys
from typing import Optional

# =========================
# 配置集中管理
# =========================
CONFIG = {
    "NO_PROXY": "127.0.0.1,localhost",
    "VLLM": {
        "HOST": "127.0.0.1",
        "PORT": 8080,
        "CUDA_VISIBLE_DEVICES": "1",
        "MODEL_PATH": "/home/ljl/Data/LLM/GLM-4-32B-0414",
        "DTYPE": "auto",
        "TENSOR_PARALLEL_SIZE": 1,
        "READY_TIMEOUT": 600,
        "PROBE_TIMEOUT": 0.1,  # 探测是否需要启动 vLLM 的短超时
    },
    "REWARD_SERVER": {
        "HOST": "127.0.0.1",
        "PORT": 38294,
    },
    "WORK_DIR": {
        "BASE_DIR": "./saves",
        "PREFIX": "ppo_reward",
        "INPUT_FILE": "server",
    },
}

# =========================
# 工具函数
# =========================
def _wait_for_port(host: str, port: int, timeout: float = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False

def _start_vllm_if_needed(cfg: dict) -> Optional[subprocess.Popen]:
    host = cfg["VLLM"]["HOST"]
    port = cfg["VLLM"]["PORT"]
    probe_timeout = cfg["VLLM"]["PROBE_TIMEOUT"]
    if _wait_for_port(host, port, timeout=probe_timeout):
        return None  # 已就绪，无需启动

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["VLLM"]["CUDA_VISIBLE_DEVICES"])

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(cfg["VLLM"]["MODEL_PATH"]),
        "--host", host,
        "--port", str(port),
        "--dtype", str(cfg["VLLM"]["DTYPE"]),
        "--tensor-parallel-size", str(cfg["VLLM"]["TENSOR_PARALLEL_SIZE"]),
    ]
    proc = subprocess.Popen(cmd, env=env)

    if not _wait_for_port(host, port, timeout=cfg["VLLM"]["READY_TIMEOUT"]):
        try:
            proc.terminate()
        except Exception:
            pass
        raise RuntimeError(f"vLLM 未在超时时间内就绪（端口 {port} 未开放）")

    return proc

def _stop_vllm(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

# =========================
# 应用入口
# =========================
def main() -> int:
    # 环境变量标准化
    os.environ["NO_PROXY"] = CONFIG["NO_PROXY"]
    os.environ["no_proxy"] = CONFIG["NO_PROXY"]

    # 启动 vLLM（若有需要）
    vllm_proc = _start_vllm_if_needed(CONFIG)

    # 启动 PPO 奖励服务器
    from llamafy import setup_save_directory
    from src.evaluation.ppo_reward_server import RewardServer

    work_dir = setup_save_directory(
        base_dir=CONFIG["WORK_DIR"]["BASE_DIR"],
        prefix=CONFIG["WORK_DIR"]["PREFIX"],
        input_file=CONFIG["WORK_DIR"]["INPUT_FILE"],
    )

    vllm_host = CONFIG["VLLM"]["HOST"]
    vllm_port = CONFIG["VLLM"]["PORT"]
    vllm_url_base = f"http://{vllm_host}:{vllm_port}/v1"

    ppo_reward_server = RewardServer(
        host=CONFIG["REWARD_SERVER"]["HOST"],
        port=CONFIG["REWARD_SERVER"]["PORT"],
        vllm_api_base=vllm_url_base,
        work_dir=work_dir,
    )

    try:
        ppo_reward_server.run()
    finally:
        _stop_vllm(vllm_proc)
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())