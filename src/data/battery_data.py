from typing import List


class BatteryData:
    def __init__(
        self,
        type: str = None,
        cycle: int = None,
        status: str = None,
        current: List[float] = None,
        voltage: List[float] = None,
        capacity: List[float] = None,
        time: List[int] = None,
    ):
        """
        Arguments:
            type: { "single", "full" }
            cycle: number of cycles
            status: { "charge", "discharge" }
            current: I
            voltage: V
            capacity: Ah
            time: second
        """
        self.type = type
        self.cycle = cycle
        self.status = status
        self.current = current
        self.voltage = voltage
        self.capacity = capacity
        self.time = time

    def to_dictionary(self):
        return {
            "type": self.type,
            "cycle": self.cycle,
            "status": self.status,
            "current": self.current,
            "voltage": self.voltage,
            "capacity": self.capacity,
            "time": self.time,
        }
