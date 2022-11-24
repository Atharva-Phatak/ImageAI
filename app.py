import lightning as L
from lightning.app.frontend import StaticWebFrontend
from lightning.app.storage import Drive
from lightning.app.utilities.frontend import AppInfo
from lightning_api_access import APIAccessFrontend
from lightning.app.api import Post
from animeGAN import AnimeGANServe


class AnimeGANFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = AnimeGANServe()

    def run(self):
        self.work.run()

    def configure_layout(self):
        return APIAccessFrontend(
            apis=[
                {
                    "name": "Generate Image",
                    "url": f"{self.work.api_url}/api/predict",
                    "method": "POST",
                    "request": {"image": "The input image."},
                    "response": {
                        "image": "data:image/png;base64,<image-actual-content>"
                    },
                }
            ]
        )


if __name__ == "__main__":
    app = L.LightningApp(AnimeGANFlow())
