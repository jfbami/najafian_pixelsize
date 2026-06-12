"""Google Drive access via a single service account across multiple owners.

Each collaborator shares their specimen folder with the service account email,
so one credential reaches every account. All access is read-only.
"""

from __future__ import annotations

import io
import random
import time
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .models import DriveFolder

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_FOLDER_MIME = "application/vnd.google-apps.folder"
_TIFF_MIMES = ("image/tiff", "image/tif")
_MAX_RETRIES = 4


class DriveClient:
    def __init__(self, service_account_json: str):
        credentials = service_account.Credentials.from_service_account_file(
            service_account_json, scopes=_SCOPES
        )
        self._service = build("drive", "v3", credentials=credentials)

    def list_specimen_folders(self, account_name: str, root_folder_id: str) -> list[DriveFolder]:
        folders: list[DriveFolder] = []
        for specimen in self._child_folders(root_folder_id):
            for acquisition in self._child_folders(specimen["id"]):
                tiffs = self._tiff_files(acquisition["id"])
                if tiffs:
                    folders.append(
                        DriveFolder(
                            account_name=account_name,
                            folder_id=acquisition["id"],
                            folder_name=acquisition["name"],
                            specimen_id=specimen["name"],
                            subfolder_path=f"{account_name}/{specimen['name']}/{acquisition['name']}",
                            frame_file_ids=[item["id"] for item in tiffs],
                            frame_count=len(tiffs),
                        )
                    )
        return folders

    def _child_folders(self, parent_id: str) -> list[dict]:
        return self._list(
            f"'{parent_id}' in parents and mimeType='{_FOLDER_MIME}' and trashed=false",
            order_by="name",
        )

    def _tiff_files(self, parent_id: str) -> list[dict]:
        mime_filter = " or ".join(f"mimeType='{mime}'" for mime in _TIFF_MIMES)
        return self._list(
            f"'{parent_id}' in parents and ({mime_filter}) and trashed=false",
            order_by="name",
        )

    def _list(self, query: str, order_by: str) -> list[dict]:
        results: list[dict] = []
        page_token = None
        while True:
            response = self._service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, size)",
                orderBy=order_by,
                pageToken=page_token,
                pageSize=1000,
            ).execute()
            results.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return results

    def download_file(self, file_id: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(_MAX_RETRIES):
            try:
                self._download_once(file_id, destination)
                return destination
            except Exception:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt + random.uniform(0.0, 1.0))
        return destination

    def _download_once(self, file_id: str, destination: Path) -> None:
        request = self._service.files().get_media(fileId=file_id)
        with io.FileIO(destination, "wb") as buffer:
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def download_thumbnail(self, file_id: str, destination: Path) -> Path:
        """Fetch Drive's server-side thumbnail; cheap relative to the full TIFF."""
        metadata = self._service.files().get(
            fileId=file_id, fields="thumbnailLink"
        ).execute()
        link = metadata.get("thumbnailLink")
        if not link:
            raise RuntimeError(f"no thumbnail available for {file_id}")
        return self._download_url(link.replace("=s220", "=s512"), destination)

    def _download_url(self, url: str, destination: Path) -> Path:
        import urllib.request

        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url)
        request.add_header(
            "Authorization", f"Bearer {self._service._http.credentials.token}"
        )
        with urllib.request.urlopen(request) as response:
            destination.write_bytes(response.read())
        return destination
