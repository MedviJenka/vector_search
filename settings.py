from pydantic_settings import BaseSettings
from pydantic import Field


# ---------------- #
#   Configuration  #
# ---------------- #

class Config(BaseSettings):
    collection_name: str = Field(default='Main', description="Collection name")
    size:            int = Field(default=100,    description="Vector size")
