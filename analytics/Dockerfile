FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir pandas numpy && useradd --create-home app
COPY minerva_data.py build_dashboard.py serve.py fetch_plotly.py ./
COPY dashboard/template.html dashboard/defaults.json dashboard/
COPY dashboard/assets dashboard/assets
COPY data data
# Bundle Plotly so the dashboard also works on networks without internet access.
RUN python fetch_plotly.py || echo "Plotly not bundled: browsers will load it from the CDN"
RUN chown -R app /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8080"]
