"""Modal entry point for the SETU public demo."""

import modal

app = modal.App("setu-public-demo")
image = modal.Image.from_dockerfile("Dockerfile")
public_demo_env = modal.Secret.from_dict(
    {"SETU_PUBLIC_DEMO": "true", "SETU_META_MODE": "cached"}
)


@app.function(image=image, secrets=[public_demo_env], min_containers=0)
@modal.concurrent(max_inputs=30)
@modal.asgi_app()
def web():
    from setu.api import app as fastapi_app

    return fastapi_app
