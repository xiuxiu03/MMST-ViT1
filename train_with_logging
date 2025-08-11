import logging
import os
from datetime import datetime

# 1. 设置日志文件名（包含时间戳）
log_dir = os.path.expanduser("~/experiment_logs")  # 自动处理 ~ 为家目录
os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
log_file = os.path.join(log_dir, f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 2. 配置日志格式
logging.basicConfig(
    level=logging.INFO,  # 记录 INFO 及以上级别的日志
    format="%(asctime)s - %(levelname)s - %(message)s",  # 时间 - 级别 - 消息
    handlers=[
        logging.FileHandler(log_file),  # 输出到文件
        logging.StreamHandler()         # 同时打印到终端
    ]
)

# 3. 记录实验超参数
logging.info("===== 实验开始 =====")
logging.info(f"时间: {datetime.now()}")
logging.info("超参数: batch_size=32, initial_lr=0.001, optimizer=Adam")

# 4. 模拟训练循环（替换为实际训练代码）
for epoch in range(5):
    # 模拟学习率衰减和损失下降
    lr = 0.001 * (0.9 ** epoch)
    loss = 1.0 / (epoch + 1)
    
    # 记录关键指标（格式化字符串对齐数据）
    logging.info(
        f"Epoch {epoch:03d} | "
        f"LR: {lr:.6f} | "
        f"Loss: {loss:.4f} | "
        f"其他指标: accuracy={epoch*0.2:.2f}"
    )

logging.info("===== 实验结束 =====")
