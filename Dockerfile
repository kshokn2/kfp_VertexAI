FROM kfp_base_image

WORKDIR /kfp

# Copy code
COPY run_kfp_vertexAI.py /kfp/

# Sets up the entry point
ENTRYPOINT [ "python", "/kfp/run_kfp_vertexAI.py"]