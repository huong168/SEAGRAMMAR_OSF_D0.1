# Reproduces the registered run exactly.
FROM python:3.11.15-slim
RUN pip install --no-cache-dir \
      scikit-learn==1.8.0 pandas==3.0.2 numpy==2.4.4 matplotlib==3.10.9
WORKDIR /seagrammar
COPY . /seagrammar
# rdkit intentionally NOT installed: the pipeline must run on the shipped
# feature table so that the registered numbers are reproduced bit for bit.
# To regenerate the tables from SMILES: pip install rdkit==2026.03.6
CMD ["bash", "RUN.sh"]
