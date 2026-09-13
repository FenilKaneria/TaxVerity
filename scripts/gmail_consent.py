"""Step 11.4c — one-time local OAuth consent, to obtain a Gmail refresh token.

Run once per Gmail sender account. Prints the refresh token to paste into
`.env` as `TAXVERITY_GMAIL_REFRESH_TOKEN`; it is never written to disk by this
script. Needs a Google Cloud OAuth client (type "Desktop app") with the Gmail
API enabled and `http://127.0.0.1:8765/` registered as a redirect URI.

Not a test target: it opens a browser and blocks on a human clicking
"Allow", the same category as Step 1.1's corpus spike.
"""

from __future__ import annotations

import json
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx2

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REDIRECT_URI = "http://127.0.0.1:8765/"
SCOPE = "https://www.googleapis.com/auth/gmail.send"


class _CallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None

    def do_GET(self) -> None:  # noqa: N802 (stdlib's own naming)
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        _CallbackHandler.code = params.get("code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Consent received. You can close this tab.")

    def log_message(self, *args: object) -> None:
        pass  # the console prompt below is the script's own status output


def main() -> None:
    client_id = input("Gmail OAuth client id: ").strip()
    client_secret = input("Gmail OAuth client secret: ").strip()

    auth_url = AUTH_URL + "?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": SCOPE,
            "access_type": "offline",
            "prompt": "consent",
        }
    )
    print(f"Opening browser for consent. If it does not open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    server = HTTPServer(("127.0.0.1", 8765), _CallbackHandler)
    server.handle_request()
    code = _CallbackHandler.code
    if code is None:
        print("No authorization code received.")
        return

    response = httpx2.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
        },
    )
    payload = response.json()
    if "refresh_token" not in payload:
        print(f"No refresh token in response: {json.dumps(payload)}")
        return
    print("\nAdd this to .env:")
    print(f"TAXVERITY_GMAIL_REFRESH_TOKEN={payload['refresh_token']}")


if __name__ == "__main__":
    main()
