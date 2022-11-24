import lightning as L
from dataclasses import dataclass
import urllib
import torch
from pathlib import Path
from animeGAN.pipeline import InferencePipeline
import io
from fastapi import UploadFile, File, Response
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import base64
from animeGAN.utils import TimeoutException
from animeGAN.constants import AppConstants


@dataclass
class AnimeGANConfig(L.BuildConfig):
    requirements = ["fastapi==0.78.0", "uvicorn==0.17.6", "torch", "numpy"]


class AnimeGANServe(L.LightningWork):
    def __init__(self, **kwargs):
        super().__init__(cloud_build_config=AnimeGANConfig(), **kwargs)
        self._model = None
        self.api_url = ""

    @staticmethod
    def _download_weights(url: str, storePath: str):
        dest = storePath / f"generator.pt"
        if not os.path.exists(dest):
            urllib.request.urlretrieve(url, dest)

    def build_pipeline(self):
        fp16 = True if torch.cuda.is_available() else False
        device = "cuda" if fp16 else "cpu"
        weights_path = Path("resources/trained_models")
        weights_path.mkdir(parents=True, exist_ok=True)
        self._download_weights(
            url="https://github.com/Atharva-Phatak/AnimeGAN/releases/download/0.0.1/generator_f_100.pt",
            storePath=weights_path,
        )
        self._model = InferencePipeline(
            weights_path=weights_path, device=device, use_fp16=fp16
        )

    def predict(self, data: bytes):
        image = Image.open(io.BytesIO(data))
        generatedImage = self._model.convertToAnime(image)
        buffered = io.BytesIO()
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return Response(content=img_str, media_type="image/png")

    def run(self):
        import subprocess
        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        if self._model is None:
            self.build_pipeline()

        app = FastAPI()
        app.POOL: ThreadPoolExecutor = None

        @app.on_event("startup")
        def startup_event():
            app.POOL = ThreadPoolExecutor(max_workers=1)

        @app.on_event("shutdown")
        def shutdown_event():
            app.POOL.shutdown(wait=False)

        @app.get("/api/health")
        def health():
            return True

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/api/predict")
        async def predict_api(data: UploadFile = File(...)):
            try:
                data = await data.read()
                result = app.POOL.submit(self.predict, data).result()
                return result
            except (TimeoutError, TimeoutException):
                raise TimeoutException()

        uvicorn.run(
            app, timeout_keep_alive=AppConstants.KEEP_ALIVE_TIMEOUT, access_log=False, loop="uvloop"
        )
