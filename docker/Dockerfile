FROM sinapsis-nvidia:base

COPY . /app/
WORKDIR /app
RUN uv sync --frozen
RUN uv pip install sinapsis-ocr[all] --extra-index-url https://pypi.sinapsis.tech