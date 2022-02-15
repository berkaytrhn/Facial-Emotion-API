from fastapi import FastAPI
from typing import Optional
import uvicorn
import sys

application = FastAPI()


@application.get("/")
async def home():
    return {"Welcome": "Page!"}


@application.get("/test")
async def testing_api(input: str, opt: Optional[int]=None):
    return {
        "Your Text": f"{input}", 
        "Opt": f"{opt}"
    }


if __name__ == "__main__":
    ip = sys.argv[1]
    port = int(sys.argv[2])

    uvicorn.run("app:application", host=ip, port=port, reload=True)
