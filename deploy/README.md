# Put the SETU demo online

These steps assume Windows 10/11 and PowerShell. The deployment serves the
frontend and API from one public URL. Set `SETU_PUBLIC_DEMO=true`; this makes
uploads use the real local video pipeline, keeps Meta in cached mode, applies
the public upload limits, and removes original videos after 24 hours.

## Before deploying

1. Install Git for Windows and Docker Desktop, then restart PowerShell.
2. Open PowerShell in the SETU folder (the folder containing `Dockerfile`).
3. Optional: put real showcase bundles in `demo_scans/<name>/`. Each folder
   needs a `result.json` and any output files it references. Do not put demo
   or fabricated measurements there.
4. Check the Docker build locally:

   ```powershell
   docker build -t setu-public-demo .
   docker run --rm -p 8000:8000 -e SETU_PUBLIC_DEMO=true -e SETU_META_MODE=cached setu-public-demo
   ```

   Open `http://localhost:8000` in a browser. Stop the local server with
   Ctrl+C.

The uploaded files and SQLite database are stored on the container's local
disk. Both hosting options below can replace containers, so this demo does not
promise durable uploads or scan history. Showcase results included in the
image return whenever a new container starts.

## Option A: Modal

1. Create an account at [modal.com](https://modal.com/) and install Python 3.11
   or newer on Windows.
2. In PowerShell, from the SETU folder, create and activate a virtual
   environment and install Modal:

   ```powershell
   py -m venv .venv-modal
   .\.venv-modal\Scripts\Activate.ps1
   python -m pip install --upgrade pip modal
   ```

3. If PowerShell blocks activation, run this once in that window and repeat
   the activation command:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

4. Sign in and deploy the included Modal app:

   ```powershell
   modal setup
   modal deploy deploy/modal_app.py
   ```

5. Modal prints the public `modal.run` URL after deployment. Open it to view
   SETU. Later, deploy an update by running the same `modal deploy` command.

The included Modal entry point builds the root `Dockerfile` and sets public
demo mode. No Meta API key is needed because live Meta requests are disabled.
See [Modal's Docker image guide](https://modal.com/docs/guide/existing-images)
and [ASGI web functions guide](https://modal.com/docs/guide/webhooks) for
platform details.

## Option B: Render

1. Put this project in a GitHub repository and push the files you want to
   deploy. Do not commit `.env` files, credentials, or private recordings.
2. Create a [Render](https://render.com/) account and choose **New** >
   **Web Service**. Connect the repository.
3. Select **Docker** as the runtime. Leave the Dockerfile path as
   `./Dockerfile` and the root directory blank if the Dockerfile is at the
   repository root.
4. Under **Environment**, add:
   - `SETU_PUBLIC_DEMO` = `true`
   - `SETU_META_MODE` = `cached`
5. Choose an instance size, then select **Create Web Service**. Render builds
   the frontend and backend image, then provides an `onrender.com` URL when
   deployment finishes.
6. Open that URL. For changes, push a commit to the connected branch and wait
   for Render to redeploy.

Render supplies `PORT`; the container binds to it automatically. See Render's
[Docker deployment guide](https://render.com/docs/docker) and
[web service guide](https://render.com/docs/web-services) for current plan and
service details.
