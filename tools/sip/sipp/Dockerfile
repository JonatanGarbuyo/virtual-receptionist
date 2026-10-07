# Deterministic SIP script runner for #25 (SIPp, pinned).
# Ubuntu 24.04 `sip-tester` package (SIPp 3.7.2). Scenarios live in
# tools/sip/sipp/scenarios/ and are versioned with the repo.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      sip-tester=1:3.7.2-1build3 \
 && rm -rf /var/lib/apt/lists
# NOTE: `sipp -v` exits non-zero by design; version is probed at runtime.
COPY scenarios/ /scenarios/
WORKDIR /scenarios
ENTRYPOINT ["sipp"]
