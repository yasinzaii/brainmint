import logging

import torch


class SimpleGPUMemoryTracker:
    """
    Track peak CUDA memory (in GB) for a chosen device.
    """
    
    def __init__(
        self,
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> None:
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.memory_records: list[tuple[int, str, float]] = []

        self.reset_peak()
    
    def reset_peak(self) -> None:
        """Reset CUDA's internal peak-memory counter."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def record_peak_memory(self, epoch: int, phase: str = "") -> str:
        """
        Capture the current CUDA peak (since last reset), log it,
        and return a human-readable string msg.
        """
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            msg = f"Epoch {epoch} {phase}: {peak_gb:.2f} GB"
            self.memory_records.append((epoch, phase, peak_gb))
            #self.logger.info(msg)
            return msg

        msg = "GPU not available"
        self.logger.warning(msg)
        return msg

    def print_all_records(self) -> None:
        """Log every recorded peak plus the overall maximum."""
        self.logger.info("=== GPU Memory Usage Records ===")
        for epoch, phase, memory_gb in self.memory_records:
            self.logger.info(f"Epoch {epoch:3d} {phase:10s}: {memory_gb:6.2f} GB")

        if self.memory_records:
            max_memory = max(self.memory_records, key=lambda x: x[2])
            self.logger.info(f"Peak Usage: Epoch {max_memory[0]} {max_memory[1]}: {max_memory[2]:.2f} GB")