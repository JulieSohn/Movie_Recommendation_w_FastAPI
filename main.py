from recommendation import get_movie_recommendation

#========================================
# API
#========================================
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates/")


@app.get("/")
async def root():
    return "Welcome to this movie recommendation engine"

@app.get("/{movie}")
async def get_recom(movie: str):
    return {"Recommended movies:": get_movie_recommendation(movie)}


@app.get("/movie/{form}")
def form_post(request: Request):
    result = "Write a movie"
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})

@app.post("/movie/{form}")
def form_post(request: Request, movie: str = Form(...)):
    result = get_movie_recommendation(movie)
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result.to_html()})


# uvicorn main:app --reload