from dataclasses import dataclass


@dataclass
class AppConstants:
    KEEP_ALIVE_TIMEOUT: int = 160
    INFERENCE_REQUEST_TIMEOUT: int = 160
