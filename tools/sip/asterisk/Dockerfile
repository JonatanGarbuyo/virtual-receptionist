# Disposable Asterisk test PBX for #25 integration (not production).
# Pinned: Ubuntu 24.04 by digest + Asterisk 20.6.0 from noble/universe.
# Provides registrar/PBX semantics only; deterministic protocol scripting
# is SIPp's job. Test-only credentials live in pjsip.conf and must never
# reach production config.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      asterisk=1:20.6.0~dfsg+~cs6.13.40431414-2build5 \
 && rm -rf /var/lib/apt/lists \
 && asterisk -V
COPY pjsip.conf extensions.conf rtp.conf /etc/asterisk/
EXPOSE 5060/udp 10000-10020/udp
CMD ["asterisk", "-f", "-U", "asterisk", "-G", "asterisk"]
