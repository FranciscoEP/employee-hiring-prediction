from pydantic import BaseSettings

class Settings(BaseSettings):
    BASE_URL: str= "https://gentle-inlet-29788.herokuapp.com"


settings = Settings()