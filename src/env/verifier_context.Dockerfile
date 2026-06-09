FROM {base_image}
WORKDIR {working_dir}
COPY . /tests/
RUN chmod +x /tests/test.sh && mkdir -p /logs/verifier /logs/artifacts
