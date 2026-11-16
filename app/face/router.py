"""HTTP-роуты лиц.

- POST /detect/face — лица в кадре или в пачке кадров (base64): бокс, пол, возраст, вектор;
- POST /embed       — вектор лица по фото (multipart, ровно одно лицо).
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.config import settings
from app.face.detector import InvalidImageError, NoSingleFaceError
from app.face.schemas import EmbedResponse, FaceBatchResponse, FaceRequest, FaceResponse

router = APIRouter(tags=["face"])


@router.post("/detect/face", response_model=FaceResponse | FaceBatchResponse)
def detect_face(payload: FaceRequest, request: Request) -> FaceResponse | FaceBatchResponse:
    """Находит лица: бокс, пол, возраст и вектор каждого.

    Принимает либо один кадр (`frame`) — ответ `{faces, count, processing_ms}`,
    либо пачку (`frames`) — ответ `{results: [{faces, count}, ...], processing_ms}`
    в том же порядке, что и кадры.
    """
    detector = request.app.state.detectors["face"]
    try:
        if payload.frames is not None:
            return FaceBatchResponse(**detector.predict_batch(payload.frames))
        return FaceResponse(**detector.predict(payload.frame))
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc) or "Invalid base64 image",
        ) from exc


@router.post("/embed", response_model=EmbedResponse)
def embed_face(
    request: Request,
    image: UploadFile = File(..., description="Фото, на нём ровно одно чёткое лицо."),
) -> EmbedResponse:
    """Вектор лица по фото. 422: `invalid image` / `no face` / `more than one face`."""
    data = image.file.read()
    if len(data) > settings.face_max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"image is larger than {settings.face_max_upload_bytes} bytes",
        )
    detector = request.app.state.detectors["face"]
    try:
        return EmbedResponse(embedding=detector.embed(data))
    except (InvalidImageError, NoSingleFaceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
