import json
from pydantic import BaseModel

class AgentResponse(BaseModel):
    movie_title: str
    movie_overview: str
    movie_platforms: list[str]
    movie_rating: float
    movie_genres: list[str]
    movie_cast: list[str]
    movie_director: str
    movie_release_date: str
    movie_languages: str
    movie_subtitles: list[str]
    movie_trailer_url: str
    movie_reason_to_watch: str
    movie_picture_url: str

class CinemaRequest(BaseModel):
    message: str
