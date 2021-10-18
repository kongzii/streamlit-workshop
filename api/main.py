import os
import shutil
import uvicorn

from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

from api.model import interactive_handler

TMP_DIR = Path('/tmp')


def create_app():
    app = FastAPI()

    app.mount("/files", StaticFiles(directory="files"), name="files")

    @app.post('/search/')
    def search(upload_file: UploadFile = File(...)):
        upload_file.file.seek(0)

        dest = TMP_DIR / upload_file.filename
        with open(dest, 'wb') as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        try:
            similar_person_images, recomm_clothes_pairs, recomm_image_paths = interactive_handler(dest)

            return {
                'error': None,
                'results': {
                    'similar_person_images': similar_person_images,
                    'recomm_clothes_pairs': recomm_clothes_pairs,
                    'recomm_image_paths': recomm_image_paths,
                },
            }

        except ValueError as e:
            if 'o valid face detecte' not in str(e):
                raise e

            return {'error': 'No face detected.', 'results': None}

    return app


if __name__ == '__main__':
    uvicorn.run(
        'api.main:create_app',
        factory=True,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        workers=1,
        reload=os.environ.get('ENV') != 'production',
        log_level='info'
    )
