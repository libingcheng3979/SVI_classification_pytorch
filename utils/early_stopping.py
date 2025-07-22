"""
早停机制
"""


class EarlyStopping:
    """
    早停类，用于监控验证损失，在损失不再下降时停止训练
    """

    def __init__(self, patience=10, verbose=True):
        """
        初始化早停实例

        Args:
            patience: 容忍验证损失不下降的轮数
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, writer=None, epoch=None):
        """
        检查是否应该触发早停

        Args:
            val_loss: 当前的验证损失
            writer: TensorBoard SummaryWriter
            epoch: 当前的训练轮次

        Returns:
            early_stop: 是否应该早停
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            # 记录早停计数器到TensorBoard
            if writer is not None and epoch is not None:
                writer.add_scalar('EarlyStopping/Counter', self.counter, epoch)

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n早停触发! 验证损失已经连续{self.patience}个epoch没有改善。")
                    print(f"最佳验证损失: {self.val_loss_min:.6f}")
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

        return self.early_stop