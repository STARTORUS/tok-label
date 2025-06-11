# TOKLABEL 部署指南

本文档提供了 TOKLABEL 的详细部署说明，包括环境要求、配置步骤和部署流程。

## 环境要求

### 硬件要求
- Kubernetes 集群

### 软件要求
- Kubernetes 1.19+
- Helm 3.0+
- Nginx Ingress Controller
- PostgreSQL 数据库
- Redis

## 配置步骤

### 1. 准备配置文件

1. 复制示例配置文件：
```bash
cp deploy/secret.example.yaml deploy/secret.yaml
cp deploy/redis.example.yaml deploy/redis.yaml
cp deploy/ls-ingress.example.yaml deploy/ls-ingress.yaml
cp deploy/ls-values.example.yaml deploy/ls-values.yaml
cp deploy/pv.example.yaml deploy/pv.yaml
```

2. 修改配置文件中的占位符：
   - `secret.yaml`: 设置数据库密码、Redis密码、API密钥等
   - `redis.yaml`: 设置Redis节点端口
   - `ls-ingress.yaml`: 设置主机名
   - `ls-values.yaml`: 设置数据库连接信息、Redis连接信息等
   - `pv.yaml`: 设置节点名称和存储路径

### 2. 创建命名空间

```bash
kubectl create namespace label-studio
```

### 3. 创建持久化存储

1. 在节点上创建必要的目录：
```bash
# 在 Label Studio 节点上
mkdir -p /mnt/data/label-studio
mkdir -p /mnt/data/ls-file-server
mkdir -p /mnt/data/redis

# 在 GPU 节点上
mkdir -p /mnt/data/huggingface-cache
mkdir -p /mnt/data/ml-backend/SAM2-image
```

2. 应用 PV 配置：
```bash
kubectl apply -f deploy/pv.yaml
```

### 4. 部署 Redis

```bash
kubectl apply -f deploy/redis.yaml
```

### 5. 部署 Label Studio

1. 添加 Label Studio Helm 仓库：
```bash
helm repo add heartex https://heartexlabs.github.io/charts/
helm repo update
```

2. 部署 Label Studio：
```bash
helm install label-studio heartex/label-studio -f deploy/ls-values.yaml -n label-studio
```

### 6. 部署 Ingress

```bash
kubectl apply -f deploy/ls-ingress.yaml
```

### 7. 部署 ML 后端服务

```bash
# 部署 SAM2 服务
kubectl apply -f deploy/SAM2.yaml

# 部署文件服务器
kubectl apply -f deploy/file-server.yaml

# 部署帧提取器
kubectl apply -f deploy/frame-extractor.yaml

# 部署特征提取服务
kubectl apply -f deploy/ip-features.yaml
kubectl apply -f deploy/discharge_timing_features.yaml
kubectl apply -f deploy/plasma-polygon.yaml
kubectl apply -f deploy/plasma-mask.yaml
```

## 验证部署

1. 检查所有 Pod 是否正常运行：
```bash
kubectl get pods -n label-studio
```

2. 检查服务是否可访问：
```bash
kubectl get svc -n label-studio
```

3. 访问 Label Studio Web 界面：
   - 打开浏览器访问 `http://<YOUR_HOST>`
   - 使用默认用户名和密码登录
   - 在 Account & Settings 中获取 API 密钥

## 配置说明

### 数据库配置
- 主机：`<DB_HOST>`
- 端口：`<DB_PORT>`
- 数据库名：`<DB_NAME>`
- 用户名：`<DB_USER>`
- 密码：在 `secret.yaml` 中配置

### Redis 配置
- 主机：`<REDIS_HOST>`
- 端口：`<REDIS_PORT>`
- 密码：在 `secret.yaml` 中配置

### Label Studio 配置
- API 密钥：在 `secret.yaml` 中配置
- 存储路径：在 `pv.yaml` 中配置
- 节点选择：在 `pv.yaml` 中配置

## 故障排除

### 常见问题

1. Pod 无法启动
   - 检查节点资源是否充足
   - 检查存储卷是否正确挂载
   - 检查配置文件是否正确

2. 服务无法访问
   - 检查 Ingress 配置
   - 检查服务端口是否正确
   - 检查网络策略

3. 数据持久化问题
   - 检查 PV 和 PVC 状态
   - 检查存储路径权限
   - 检查存储类配置

### 日志查看

```bash
# 查看 Label Studio 日志
kubectl logs -f deployment/label-studio-ls-app -n label-studio

# 查看 Redis 日志
kubectl logs -f deployment/redis -n label-studio

# 查看 ML 后端日志
kubectl logs -f deployment/ml-backend-sam2 -n label-studio
```

## 维护和更新

### 更新 Label Studio

```bash
helm upgrade label-studio heartex/label-studio -f deploy/ls-values.yaml -n label-studio
```

### 备份数据

1. 备份数据库：
```bash
kubectl exec -it <postgres-pod> -n label-studio -- pg_dump -U <username> <database> > backup.sql
```

2. 备份 Redis 数据：
```bash
kubectl exec -it <redis-pod> -n label-studio -- redis-cli SAVE
kubectl cp label-studio/<redis-pod>:/data/dump.rdb ./redis-backup.rdb
```

### 恢复数据

1. 恢复数据库：
```bash
kubectl exec -i <postgres-pod> -n label-studio -- psql -U <username> <database> < backup.sql
```

2. 恢复 Redis 数据：
```bash
kubectl cp ./redis-backup.rdb label-studio/<redis-pod>:/data/dump.rdb
kubectl exec -it <redis-pod> -n label-studio -- redis-cli BGREWRITEAOF
```

## 安全建议

1. 定期更新密码和密钥
2. 使用 HTTPS 访问 Label Studio
3. 限制数据库和 Redis 的访问范围
4. 定期备份数据
5. 监控系统日志和资源使用情况 