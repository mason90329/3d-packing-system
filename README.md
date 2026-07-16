# 3D Packing System

Streamlit-based 3D packing web app.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload these files:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`
2. Go to <https://share.streamlit.io/>.
3. Connect GitHub and select the repository.
4. Set the main file path to `app.py`.
5. Deploy.

After deployment, Streamlit will provide a public URL that can be opened on phones, tablets, and other computers.

## Deploy to Render

1. Create a GitHub repository and upload all project files.
2. Go to <https://render.com/>.
3. Create a new Web Service from the GitHub repository.
4. Render can use `render.yaml` automatically, or use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
5. Deploy and open the generated Render URL.
