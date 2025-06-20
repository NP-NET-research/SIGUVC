from torch.utils.tensorboard import SummaryWriter
import logging

class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_stats(self, step: int, stats: dict):
        self.step = step
        policy_loss = stats.get("ppo/loss/policy", 0)
        value_loss  = stats.get("ppo/loss/value", 0)
        total_loss  = stats.get("ppo/loss/total", 0)
        kl_div      = stats.get("objective/kl", 0)
        advantage   = stats.get("ppo/policy/advantages_mean", 0)
        reward = stats.get("custom/reward", 0)
        wordbank    = stats.get("custom/wordbank", 0)
        semantics   = stats.get("custom/semantics", 0)
        expression  = stats.get("custom/expression", 0)

        self.writer.add_scalar("Loss/Policy",    policy_loss, step)
        self.writer.add_scalar("Loss/Value",     value_loss,  step)
        self.writer.add_scalar("Loss/Total",     total_loss,  step)
        self.writer.add_scalar("Metrics/KL",     kl_div,      step)
        self.writer.add_scalar("Metrics/Adv",    advantage,   step)
        self.writer.add_scalar("Reward/Average", reward,      step)
        self.writer.add_scalar("Reward/Wordbank", wordbank,   step)
        self.writer.add_scalar("Reward/Semantics", semantics, step)
        self.writer.add_scalar("Reward/Expression", expression, step)


    def close(self):
        self.writer.close()

    def setup_logger(self, log_file: str):
        logger = logging.getLogger("train")
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
