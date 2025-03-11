from pathlib import Path
import pyprojroot
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv, load_dotenv

class PathInfo:
    """
    Base information class that defines core paths and environment settings.
    These paths are crucial for the application to locate resources and configurations.
    """
    HOME: Path = Path.home()
    BASE: Path = pyprojroot.find_root(pyprojroot.has_dir("config"))
    WORKSPACE: Path = BASE.parent.parent
    ENV = "dev"

load_dotenv(Path(PathInfo.BASE, ".env"))

class GeneralSettings(BaseSettings, PathInfo):
    """
    Main application settings class that combines environment variables and default configurations.
    This class is used throughout the application to access configuration values.
    """
    model_config = SettingsConfigDict(case_sensitive=True)
    DEBUG: bool = False

class TrainingSettings(BaseSettings, PathInfo):
    """
    Settings related to model training configuration.
    """
    model_config = SettingsConfigDict(case_sensitive=True)
    
    # Data paths
    DATA_DIR: Path = Path(PathInfo.BASE, "data/places365")
    OUTPUT_DIR: Path = Path(PathInfo.BASE, "outputs")
    
    # Training parameters
    BATCH_SIZE: int = 16
    VAL_BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    NUM_EPOCHS: int = 10
    IMAGE_SIZE: int = 384
    
    # Distillation parameters
    GAMMA: float = 0.5
    TEMPERATURE: float = 0.5
    
    # Model configurations
    TEACHER_MODEL: str = "depth-anything/Depth-Anything-V2-Large-hf"
    STUDENT_MODEL: str = "facebook/dpt-dinov2-small-kitti"
    
    # Training logistics
    SEED: int = 42
    SAVE_STEPS: int = 1000
    EVAL_STEPS: int = 500
    LOGGING_STEPS: int = 100
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 3
    EARLY_STOPPING_THRESHOLD: float = 0.01
    
    # Optimizer settings
    WARMUP_STEPS: int = 500
    LR_SCHEDULER_TYPE: str = "cosine"
    MIN_LR_RATIO: float = 0.1
    