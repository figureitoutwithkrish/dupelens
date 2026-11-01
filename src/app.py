"""Local web app: paste or upload a new listing, get "already listed?" + price check.

Serves the same dashboard as the hosted site, plus a live /api/check endpoint
that runs the trained models on the listing you give it.
Run: python -m src.app   then open http://127.0.0.1:8000
"""
import threading

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.service import Matcher

DASHBOARD = config.ROOT / "dashboard"
app = FastAPI(title="DupeLens")
matcher = Matcher()
# Load the two encoders in the background so the first check does not wait ~10 s.
_warm = config.INDIA_IMAGES / f"{matcher.catalog['listing_id'][0]}.jpg"
threading.Thread(target=lambda: matcher.encode("warm up", _warm.read_bytes()), daemon=True).start()


@app.get("/")
def index():
    return FileResponse(DASHBOARD / "index.html")


@app.get("/api/health")
def health():
    return {"live": True, "catalog": len(matcher.catalog), "category": config.INDIA_CATEGORY["amazon"]}


@app.post("/api/check")
async def check(title: str = Form(...), price: float | None = Form(None),
                image_url: str | None = Form(None), image: UploadFile | None = File(None)):
    """Title + (photo file or photo URL) [+ price] -> verdict, matching listings, typical price."""
    data = await image.read() if image is not None and image.filename else None
    if not title.strip() or (data is None and not image_url):
        raise HTTPException(400, "Give a title and a photo (upload or image URL).")
    try:
        return matcher.check_new(title.strip(), image_bytes=data, image_url=image_url, price=price)
    except Exception as err:  # bad image URL, unreadable file
        raise HTTPException(422, f"Could not read that photo: {err}")


app.mount("/data", StaticFiles(directory=DASHBOARD / "data"), name="data")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
