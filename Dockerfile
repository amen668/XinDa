# Runtime image for the foundation-translator v3 pipeline (Acquire→Convert→Extract).
FROM python:3.12-slim

# LaTeXML toolkit (latexmlc/latexmlpost) + a modest TeX set so most arXiv
# papers convert, plus ImageMagick. texlive-latex-extra covers common classes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        latexml \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-fonts-recommended \
        texlive-fonts-extra \
        texlive-latex-extra \
        texlive-science \
        texlive-publishers \
        texlive-lang-english \
        texlive-plain-generic \
        imagemagick \
        ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Default: run the extract smoke test on one paper. Override via
#   docker compose run --rm app python -m xinda.cli.<other>
CMD ["python", "-m", "xinda.cli.extract_smoke", "2503.15129"]
