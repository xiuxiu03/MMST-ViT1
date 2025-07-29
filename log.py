import logging
from datetime import datetime

# 1. 配置日志系统
log_file = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 保存到文件
        logging.StreamHandler()         # 同时打印到控制台
    ]
)

# 2. 在代码中记录关键信息
logging.info(f"Starting training with lr={lr}, batch_size={batch_size}")
try:
    model.train()
    for epoch in range(epochs):
        loss = train_one_epoch()
        logging.info(f"Epoch {epoch}: Loss={loss:.4f}")  # 自动添加时间戳
except Exception as e:
    logging.error(f"Training failed: {str(e)}", exc_info=True)  # 记录完整错误堆栈
