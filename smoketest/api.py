import requests
from urllib.parse import urljoin


class API:
    def __init__(self, user, password, server):
        self._baseurl = server
        self._loginname = user
        self._token = self._get_jwt(password, "user")
        self._auth_header = {"Authorization": self._token}

    def _get_jwt(self, password, logintype):
        nameProp = logintype + "name"

        url = urljoin(self._baseurl, "/authenticate_" + logintype)
        r = requests.post(url, data={nameProp: self._loginname, "password": password})

        if r.status_code == 200:
            return r.json().get("token")
        elif r.status_code == 422:
            raise ValueError(
                "Could not log on as '{}'.  Please check {} name.".format(
                    self._loginname, logintype
                )
            )
        elif r.status_code == 401:
            raise ValueError(
                "Could not log on as '{}'.  Please check password.".format(
                    self._loginname
                )
            )
        else:
            r.raise_for_status()

    def _check_response(self, r):
        if r.status_code in (400, 422):
            j = r.json()
            messages = j.get("messages", j.get("message", ""))
            raise IOError("request failed ({}): {}".format(r.status_code, messages))
        r.raise_for_status()
        return r.json()

    def download_raw(self, recording_id):
        return self._download_recording(recording_id, "downloadRawJWT")

    def query_rec(self, recording_id):
        url = urljoin(self._baseurl, "/api/v1/recordings/{}".format(recording_id))
        r = requests.get(url, headers=self._auth_header)
        return self._check_response(r)

    def _download_recording(self, recording_id, jwt_key):
        d = self.query_rec(recording_id)
        return self._download_signed(d[jwt_key])

    def _download_signed(self, token):
        r = requests.get(
            urljoin(self._baseurl, "/api/v1/signedUrl"),
            params={"jwt": token},
            stream=True,
        )
        r.raise_for_status()
        yield from r.iter_content(chunk_size=4096)

    def get_tracks(self, recording_id):
        url = urljoin(
            self._baseurl, "/api/v1/recordings/{}/tracks".format(recording_id)
        )
        r = requests.get(url, headers=self._auth_header)
        return self._check_response(r)

    def save_file(self, filename, url):
        return iter_to_file(filename, url)


def iter_to_file(filename, source, overwrite=True):
    if not overwrite and Path(filename).is_file():
        print("{} already exists".format(filename))
        return False
    with open(filename, "wb") as f:
        for chunk in source:
            f.write(chunk)
    return True
