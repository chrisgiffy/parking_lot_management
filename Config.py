from mrcnn.config import Config


class ParkingLotConfig(Config):
    NAME = "parkinglot"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # background + 3 shapes
    STEPS_PER_EPOCH = 100
    # GPU_COUNT = 3


config = ParkingLotConfig()
config.display()