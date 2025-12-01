## 运行方式

```bash
# 1) 启动 RLENV 控制端与资源池
cd /root/autodl-tmp/rl/Deepin-RLENV
python start.py all_config.yaml

# 2) 设置环境变量（按需）
export PYTHONPATH=/root/autodl-tmp/rl/AReaL
export CONTROLLER_BASE_URL=http://127.0.0.1:8081
# 如 RLENV 开启鉴权：
# export ENV_WORKER_TOKEN=your_token

# 3) 运行训练（两种启动方式任选其一）

# 3.1 本地单机（local）方式
cd /root/autodl-tmp/rl
python3 -m areal.launcher.local \
  AReaL/examples/multimodel_multiturn/train_qwen25vl.py \
  --config AReaL/examples/multimodel_multiturn/config.yaml \
  experiment_name=qwen25vl-multimodel-multiturn \
  trial_name=trial0 \
  allocation_mode=sglang:d4p1t1+d4p1t1 \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=8

# 3.2 分布式（ray）方式
cd /root/autodl-tmp/rl
python3 -m areal.launcher.ray \
  AReaL/examples/multimodel_multiturn/train_qwen25vl.py \
  --config AReaL/examples/multimodel_multiturn/config.yaml \
  experiment_name=qwen25vl-multimodel-multiturn \
  trial_name=trial0 \
  allocation_mode=sglang:d4p1t1+d4p1t1 \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=8

# 3.2.1 多节点（ray 集群）示例
# 先在各节点上启动 Ray 集群：
#   Head 节点：
#     ray start --head --port=6379
#   Worker 节点：
#     ray start --address <head_ip>:6379
# 确认所有节点已加入：
#   在 head 节点执行：ray status
# 然后在 head 节点运行训练：
cd /root/autodl-tmp/rl
python3 -m areal.launcher.ray \
  AReaL/examples/multimodel_multiturn/train_qwen25vl.py \
  --config AReaL/examples/multimodel_multiturn/config.yaml \
  experiment_name=qwen25vl-multimodel-multiturn \
  trial_name=trial0 \
  allocation_mode=sglang:d8p1t1+d8p1t1 \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=8

# 说明：
# - 命令中无需填写每个节点 IP，Ray 启动器会连接已启动的 Ray 集群并通过 Placement Group 分配资源。
# - 请确保 cluster.n_nodes 与集群节点数一致，cluster.n_gpus_per_node 与每节点实际 GPU 数一致。
# - allocation_mode 应与资源规模匹配，例如 d8 表示 8 个数据并行实例，可分布在 2 个节点的 16 块 GPU 上根据具体设置调整。
```
