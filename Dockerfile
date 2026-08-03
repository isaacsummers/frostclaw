FROM node:22-alpine

WORKDIR /app

# Copy the standalone proxy script — zero npm dependencies, node builtins only.
COPY snowflake-proxy.mjs .

# The original script binds to 127.0.0.1 (correct for host, wrong for Docker).
# Patch listen() and the log message. The grep at the end confirms the patch landed.
RUN sed -i \
  's/server\.listen(port, "127\.0\.0\.1"/server.listen(port, "0.0.0.0"/' \
  snowflake-proxy.mjs && \
  sed -i \
  's|listening on http://127\.0\.0\.1|listening on http://0.0.0.0|' \
  snowflake-proxy.mjs && \
  grep 'server.listen' snowflake-proxy.mjs

EXPOSE 18790

CMD ["node", "snowflake-proxy.mjs"]
