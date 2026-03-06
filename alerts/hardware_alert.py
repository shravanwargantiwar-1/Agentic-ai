class HardwareAlert:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def trigger(self, message: str) -> None:
        if self.enabled:
            print(f"[HARDWARE ALERT] {message}")
