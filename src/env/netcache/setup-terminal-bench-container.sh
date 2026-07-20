#!/usr/bin/env bash
# Configures optional host caches inside each Terminal-Bench task container to
# avoid repeated apt/uv/pip downloads. Missing caches, minimal images, or
# read-only config fall back to bounded upstream access unless
# FRAMEWORK_REQUIRE_CACHES=1.

cache_required=false
if [ "${FRAMEWORK_REQUIRE_CACHES:-0}" = "1" ]; then
  cache_required=true
fi
network_freeze_enabled=false
proxy_auth=""
if [ -n "${FRAMEWORK_NETWORK_CACHE_SCOPE:-}" ]; then
  network_freeze_enabled=true
  proxy_auth="framework:${FRAMEWORK_NETWORK_CACHE_SCOPE}@"
fi

require_cache() {
  name="$1"
  port="$2"
  if timeout 2 bash -c ": < /dev/tcp/host.docker.internal/${port}" 2>/dev/null; then
    return 0
  fi
  if [ "$cache_required" = true ]; then
    printf 'required framework cache unavailable: %s on host.docker.internal:%s\n' \
      "$name" "$port" >&2
    exit 1
  fi
  return 1
}

apt_conf=/etc/apt/apt.conf.d/00-framework-apt-network
mkdir -p /etc/apt/apt.conf.d 2>/dev/null || true
cat > "$apt_conf" 2>/dev/null <<'EOF' || true
Acquire::Retries "2";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
// Get:/Fetched lines carry run-varying download rates that fork otherwise-
// identical trajectories; level 1 still prints them. Measured side effect:
// level 2 implies assume-yes.
quiet "2";
EOF

# Quiet run-varying progress output for commands run via bash -lc.
mkdir -p /etc/profile.d 2>/dev/null || true
cat > /etc/profile.d/framework-quiet-progress.sh 2>/dev/null <<'EOF' || true
export UV_NO_PROGRESS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_VERBOSITY=error
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
EOF
if [ ! -e /root/.curlrc ]; then
  cat > /root/.curlrc 2>/dev/null <<'EOF' || true
silent
show-error
EOF
fi
mkdir -p /etc 2>/dev/null || true
cat > /etc/pip.conf 2>/dev/null <<'EOF' || true
[global]
progress_bar = off
# The default 15s trips on a busy cache and prints run-varying retry WARNINGs.
timeout = 60
EOF

# Route apt through the per-scope freeze proxy; shared revalidating caches drift
# between recording and replay. HTTP needs no CA; auth carries the scope.
if [ "$network_freeze_enabled" = true ] && require_cache https-proxy 3144; then
  cat >> "$apt_conf" 2>/dev/null <<EOF || true
Acquire::http::Proxy "http://${proxy_auth}host.docker.internal:3144";
Acquire::https::Proxy "http://${proxy_auth}host.docker.internal:3144";
// Frozen InRelease bytes outlive their Valid-Until window by design; a
// time-validity check contradicts a first-write-wins snapshot.
Acquire::Check-Valid-Until "false";
// apt pipelines up to 10 requests per connection; proxies commonly
// mis-sequence pipelined responses, and mitmproxy is no exception.
Acquire::http::Pipeline-Depth "0";
EOF
fi

# force-unsafe-io is acceptable: crashed task containers are discarded.
dpkg_conf=/etc/dpkg/dpkg.cfg.d/00-framework-unsafe-io
mkdir -p /etc/dpkg/dpkg.cfg.d 2>/dev/null || true
printf 'force-unsafe-io\n' > "$dpkg_conf" 2>/dev/null || true

# Minimal task images may lack curl/wget/python, so the CA download speaks
# HTTP over bash's /dev/tcp. Only correct for a server that honors
# `Connection: close` and ends the body by closing the socket (python -m
# http.server does).
fetch_http_file() {
  host="$1"
  port="$2"
  path="$3"
  target="$4"
  tmp="${target}.tmp"
  rm -f "$tmp"
  exec 3<>"/dev/tcp/${host}/${port}" || return 1
  printf 'GET %s HTTP/1.1\r\nHost: %s:%s\r\nConnection: close\r\n\r\n' \
    "$path" "$host" "$port" >&3 || {
    exec 3<&-
    exec 3>&-
    return 1
  }
  IFS= read -r status <&3 || {
    exec 3<&-
    exec 3>&-
    return 1
  }
  case "$status" in
    *" 200 "*) ;;
    *)
      exec 3<&-
      exec 3>&-
      return 1
      ;;
  esac
  while IFS= read -r header <&3; do
    header="${header%$'\r'}"
    [ -z "$header" ] && break
  done
  cat <&3 > "$tmp" || {
    rm -f "$tmp"
    exec 3<&-
    exec 3>&-
    return 1
  }
  exec 3<&-
  exec 3>&-
  [ -s "$tmp" ] || {
    rm -f "$tmp"
    return 1
  }
  mv "$tmp" "$target"
}

# Proxy settings and the CA are agent-visible but deterministic across runs.
# Probe distro trust-store paths before export: a nonexistent bundle breaks all
# HTTPS. Export explicit bundle variables for certifi clients; profile.d applies
# because every framework-issued command runs through `bash -lc`.
if [ "$network_freeze_enabled" = true ] &&
  require_cache https-proxy 3144 &&
  require_cache https-ca 3145; then
  ca_path=/usr/local/share/ca-certificates/framework-https-cache.crt
  rhel_ca_path=/etc/pki/ca-trust/source/anchors/framework-https-cache.crt
  mkdir -p /usr/local/share/ca-certificates /etc/profile.d 2>/dev/null || true
  if fetch_http_file \
    host.docker.internal \
    3145 \
    /mitmproxy-ca-cert.pem \
    "$ca_path" 2>/dev/null; then
    ca_installed=false
    ca_bundle=""
    if command -v update-ca-certificates >/dev/null 2>&1 &&
      update-ca-certificates >/dev/null 2>&1; then
      ca_installed=true
      ca_bundle=/etc/ssl/certs/ca-certificates.crt
    elif command -v update-ca-trust >/dev/null 2>&1 &&
      mkdir -p /etc/pki/ca-trust/source/anchors 2>/dev/null &&
      cp "$ca_path" "$rhel_ca_path" 2>/dev/null &&
      update-ca-trust extract >/dev/null 2>&1; then
      ca_installed=true
      if [ -f /etc/pki/tls/certs/ca-bundle.crt ]; then
        ca_bundle=/etc/pki/tls/certs/ca-bundle.crt
      elif [ -f /etc/ssl/certs/ca-certificates.crt ]; then
        ca_bundle=/etc/ssl/certs/ca-certificates.crt
      fi
    fi
    if [ "$ca_installed" != true ] || [ -z "$ca_bundle" ]; then
      ca_bundle=/etc/ssl/certs/ca-certificates.crt
      mkdir -p /etc/ssl/certs 2>/dev/null || true
      if [ -f "$ca_bundle" ]; then
        cat "$ca_path" >> "$ca_bundle" 2>/dev/null && ca_installed=true
      else
        cp "$ca_path" "$ca_bundle" 2>/dev/null && ca_installed=true
      fi
    fi
    if [ "$ca_installed" = true ]; then
      cat > /etc/profile.d/framework-https-proxy.sh 2>/dev/null <<EOF || true
export http_proxy=http://${proxy_auth}host.docker.internal:3144
export https_proxy=http://${proxy_auth}host.docker.internal:3144
export HTTP_PROXY=http://${proxy_auth}host.docker.internal:3144
export HTTPS_PROXY=http://${proxy_auth}host.docker.internal:3144
export no_proxy=localhost,127.0.0.1,::1,host.docker.internal
export NO_PROXY=localhost,127.0.0.1,::1,host.docker.internal
export SSL_CERT_FILE=$ca_bundle
export REQUESTS_CA_BUNDLE=$ca_bundle
export CURL_CA_BUNDLE=$ca_bundle
export GIT_SSL_CAINFO=$ca_bundle
export NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/framework-https-cache.crt
EOF
    elif [ "$cache_required" = true ]; then
      printf 'required framework https cache CA could not be installed\n' >&2
      exit 1
    fi
  elif [ "$cache_required" = true ]; then
    printf 'required framework https cache CA could not be fetched\n' >&2
    exit 1
  fi
fi

# Once this is exported uv treats the mirror as authoritative and will not
# fall back upstream on a later failure, so it is only set when the mirror is
# answering right now.
if require_cache uv-python-mirror 3143; then
  mkdir -p /etc/profile.d 2>/dev/null || true
  cat > /etc/profile.d/framework-uv-mirror.sh 2>/dev/null <<'EOF' || true
export UV_PYTHON_INSTALL_MIRROR=http://host.docker.internal:3143
EOF
fi

if require_cache pypi 3141; then
  mkdir -p /etc/uv 2>/dev/null || true
  cat > /etc/uv/uv.toml 2>/dev/null <<'EOF' || true
[[index]]
url = "http://host.docker.internal:3141/index/"
default = true
EOF
  cat >> /etc/pip.conf 2>/dev/null <<'EOF' || true
index-url = http://host.docker.internal:3141/index/
trusted-host = host.docker.internal
EOF
fi

# The fakerandom shim (/opt/framework/libfaketimeMT.so.1) is bind-mounted read-only
# by the framework for pin_urandom tasks, so no per-trial apt install runs here.

true
