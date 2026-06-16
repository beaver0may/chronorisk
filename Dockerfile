FROM pytorch/pytorch:2.9.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./pyproject.toml
COPY chronorisk ./chronorisk
COPY legs ./legs
RUN pip install --no-cache-dir .

ENTRYPOINT ["chronorisk"]
CMD ["chart"]
