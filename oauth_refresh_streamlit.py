import json
import time
import requests
import streamlit as st

from pathlib import Path

st.set_page_config(page_title="OAuth Refresh Token", page_icon="🔐", layout="centered")
st.title("🔐 Générer un refresh_token Google Drive (via Streamlit)")

CLIENT_SECRET_PATH = Path("client_secret.json")  # à la racine
SCOPES = ["https://www.googleapis.com/auth/drive"]  # lecture+écriture

def _load_client_secret(path: Path):
    if not path.exists():
        st.error(f"Fichier introuvable: {path.resolve()}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Supporte "installed" ou "web"
    if "installed" in data:
        cfg = data["installed"]
    elif "web" in data:
        cfg = data["web"]
    else:
        st.error("client_secret.json invalide (clé 'installed' ou 'web' absente).")
        st.stop()
    return cfg

cfg = _load_client_secret(CLIENT_SECRET_PATH)
client_id = cfg.get("client_id", "")
client_secret = cfg.get("client_secret", "")
token_uri = cfg.get("token_uri", "https://oauth2.googleapis.com/token")

st.caption(f"client_id: {client_id[:12]}…  | token_uri: {token_uri}")

tab_local, tab_cloud = st.tabs(["✅ Mode LOCAL (recommandé)", "☁️ Mode CLOUD (Device Flow)"])

# -------------------------
# MODE LOCAL — loopback
# -------------------------
with tab_local:
    st.subheader("Mode LOCAL (streamlit run sur ton ordi)")
    st.write(
        "Ça ouvre Google dans ton navigateur, tu acceptes, puis l’app affiche le `refresh_token`.\n\n"
        "➡️ À utiliser sur ton Mac/PC (pas Streamlit Cloud)."
    )

    if st.button("🚀 Générer refresh_token (LOCAL)"):
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_PATH),
                scopes=SCOPES,
            )
            # IMPORTANT: prompt='consent' + access_type='offline' pour forcer refresh_token
            creds = flow.run_local_server(
                port=0,
                prompt="consent",
                access_type="offline",
                include_granted_scopes="true",
            )

            st.success("✅ OK — refresh_token généré")
            st.code(creds.refresh_token or "(vide)", language="text")
            st.info(
                "Copie ce refresh_token dans Streamlit Secrets:\n"
                "[gdrive_oauth] refresh_token = \"...\""
            )
        except Exception as e:
            st.error(f"Erreur: {e}")

# -------------------------
# MODE CLOUD — Device Flow
# -------------------------
with tab_cloud:
    st.subheader("Device Flow (marche sur Streamlit Cloud)")
    st.write(
        "Tu obtiens un **code** + un lien Google. Tu autorises l’app, puis on poll le token endpoint.\n"
        "➡️ Utile si tu veux générer le refresh_token sans exécuter en local."
    )

    if not client_id or not client_secret:
        st.warning("client_id / client_secret manquants dans client_secret.json.")
    else:
        if "device_flow" not in st.session_state:
            st.session_state.device_flow = None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔑 Démarrer Device Flow"):
                try:
                    r = requests.post(
                        "https://oauth2.googleapis.com/device/code",
                        data={
                            "client_id": client_id,
                            "scope": " ".join(SCOPES),
                        },
                        timeout=30,
                    )
                    r.raise_for_status()
                    st.session_state.device_flow = r.json()
                except Exception as e:
                    st.error(f"Erreur device/code: {e}")

        with col2:
            if st.button("🧹 Reset"):
                st.session_state.device_flow = None

        df = st.session_state.device_flow
        if df:
            st.success("1) Va sur le lien et entre le code")
            st.write("🔗 Verification URL:")
            st.code(df.get("verification_url") or df.get("verification_uri") or "", language="text")
            st.write("🔢 Code:")
            st.code(df.get("user_code", ""), language="text")

            st.caption("2) Après avoir autorisé, clique sur “Récupérer token” ci-dessous.")

            if st.button("📥 Récupérer token (poll)"):
                device_code = df.get("device_code")
                interval = int(df.get("interval", 5))
                expires_in = int(df.get("expires_in", 1800))

                start = time.time()
                last_err = None

                while time.time() - start < expires_in:
                    try:
                        tr = requests.post(
                            token_uri,
                            data={
                                "client_id": client_id,
                                "client_secret": client_secret,
                                "device_code": device_code,
                                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                            },
                            timeout=30,
                        )
                        data = tr.json()

                        # Success
                        if tr.status_code == 200 and "access_token" in data:
                            refresh_token = data.get("refresh_token", "")
                            st.success("✅ Token reçu")
                            st.write("refresh_token:")
                            st.code(refresh_token or "(VIDE — Google n’a pas renvoyé de refresh_token)", language="text")

                            if not refresh_token:
                                st.warning(
                                    "Google n’a pas renvoyé de refresh_token. "
                                    "Ça peut arriver si tu avais déjà autorisé l’app.\n\n"
                                    "✅ Solution: révoque l’accès ici puis réessaie:\n"
                                    "https://myaccount.google.com/permissions\n\n"
                                    "Et assure-toi qu’on force bien 'offline' (device flow ne le garantit pas toujours). "
                                    "Dans ce cas, le mode LOCAL est le plus fiable."
                                )
                            else:
                                st.info("Copie-le dans Streamlit Secrets sous [gdrive_oauth].")
                            break

                        # Pending
                        err = data.get("error")
                        if err in ("authorization_pending", "slow_down"):
                            last_err = err
                            time.sleep(interval + (3 if err == "slow_down" else 0))
                            continue

                        # Other errors
                        st.error(f"Erreur token: {data}")
                        break

                    except Exception as e:
                        last_err = str(e)
                        time.sleep(interval)

                else:
                    st.error(f"Timeout device flow (dernier état: {last_err})")
        else:
            st.info("Clique “Démarrer Device Flow” pour générer un code.")
