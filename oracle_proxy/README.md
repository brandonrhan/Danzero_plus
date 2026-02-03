# Oracle Proxy Server

这是一个 WebSocket 代理服务器，用于在原始 guandan 服务器和 bot 客户端之间注入 oracle 信息（所有玩家的手牌）。

## 架构

```
Bot 0  ←┐
Bot 1  ←┼→  Proxy Server (23457)  ←→  Original Server (23456)
Bot 2  ←┤
Bot 3  ←┘
           ↓
     收集所有玩家手牌
     注入到 act 消息中
```

## 工作原理

1. **beginning 阶段**：代理收集每个玩家的初始手牌
2. **play 阶段**：代理跟踪打出的牌，更新各玩家手牌
3. **act 消息**：代理注入 `allHandCards` 字段，包含所有玩家手牌

## 消息格式变化

原始 `act` 消息：
```json
{
    "type": "act",
    "stage": "play",
    "handCards": ["H2", "S3", ...],  // 只有当前玩家的手牌
    ...
}
```

代理后的 `act` 消息：
```json
{
    "type": "act",
    "stage": "play",
    "handCards": ["H2", "S3", ...],  // 当前玩家的手牌
    "allHandCards": [                 // 所有玩家的手牌
        ["H2", "S3", ...],  // Position 0
        ["D4", "C5", ...],  // Position 1
        ["HA", "SK", ...],  // Position 2
        ["D7", "C8", ...]   // Position 3
    ],
    "oracleEnabled": true,
    ...
}
```

## 使用方法

### 1. 启动原始服务器
```bash
cd /mnt/c/Users/admin/Documents/GitHub/Danzero_plus/actor_n
./guandan &
```

### 2. 启动代理服务器
```bash
cd /mnt/c/Users/admin/Documents/GitHub/danzero_oracle/oracle_proxy
bash run_proxy.sh
```

### 3. Bot 连接到代理
Bot 连接到 `ws://localhost:23457` 而不是 `ws://localhost:23456`

## 配置

环境变量：
- `PROXY_PORT`: 代理监听端口 (默认: 23457)
- `UPSTREAM_HOST`: 原始服务器地址 (默认: localhost)
- `UPSTREAM_PORT`: 原始服务器端口 (默认: 23456)

命令行参数：
```bash
python3 proxy_server.py --proxy-port 23457 --upstream-host localhost --upstream-port 23456 --debug
```

## 注意事项

1. 代理需要在原始服务器启动后运行
2. 所有 4 个 bot 必须连接到代理（而不是原始服务器）
3. 代理会按连接顺序分配位置 (0, 1, 2, 3)
