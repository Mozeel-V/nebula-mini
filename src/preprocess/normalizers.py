import re

HASH_MD5 = re.compile(r"\b[a-fA-F0-9]{32}\b")
HASH_SHA1 = re.compile(r"\b[a-fA-F0-9]{40}\b")
HASH_SHA256 = re.compile(r"\b[a-fA-F0-9]{64}\b")

IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.|$)){4}\b")
IPV6 = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){1,7}[A-Fa-f0-9]{1,4}\b")
DOMAIN = re.compile(r"\b([a-zA-Z0-9-]+\.)+(com|net|org|info|io|co|ru|in|uk|edu|biz)\b", re.IGNORECASE)

DRIVE = re.compile(r"^[A-Za-z]:\\")
USERSEG = re.compile(r"\\Users\\[^\\]+\\", re.IGNORECASE)

def norm_hash(s: str) -> str:
    s = HASH_SHA256.sub("<sha256>", s)
    s = HASH_SHA1.sub("<sha1>", s)
    s = HASH_MD5.sub("<md5>", s)
    return s

def classify_ipv4(ip: str) -> str:
    try:
        parts = [int(p) for p in ip.split('.')]
        if ip.startswith("127."):
            return "<ip_loopback>"
        if parts[0] == 10 or (parts[0] == 192 and parts[1] == 168) or (parts[0] == 172 and 16 <= parts[1] <= 31):
            return "<ip_private>"
    except Exception:
        pass
    return "<ip_public>"

def norm_ips_domains_paths(s: str) -> str:
    # IPv6 first
    s = IPV6.sub("<ip_ipv6>", s)

    # IPv4 with class
    def _ipv4_repl(m):
        return classify_ipv4(m.group(0))
    s = IPV4.sub(_ipv4_repl, s)

    # Domains
    s = DOMAIN.sub("<domain>", s)

    # Windows drive and user
    s = DRIVE.sub("<drive>\\", s)
    s = USERSEG.sub("\\Users\\<user>\\", s)
    return s

def normalize(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = norm_hash(value)
    value = norm_ips_domains_paths(value)
    return value
