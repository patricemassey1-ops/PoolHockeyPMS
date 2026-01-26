from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]

flow = InstalledAppFlow.from_client_secrets_file(
    "client_secret.json",  # fichier téléchargé depuis Google Cloud
    SCOPES
)

creds = flow.run_local_server(port=0)

print("REFRESH TOKEN:")
print(creds.refresh_token)
