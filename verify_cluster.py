"""Cluster Health Verifier for Singularis v5.0

Checks the live status of the 4-device SEPHIROT cluster:

- Cygnus (AMD 2x7900XT): 10 expert endpoints (1234–1243)
- MacBook Pro M3: MoE (2000) + AURA-Brain (3000)
- NVIDIA Laptop (RTX 5060): Positronic Network (4000)

Usage:
    python verify_cluster.py

Optional overrides via CLI:
    python verify_cluster.py --cygnus-ip 192.168.1.50 --macbook-ip 192.168.1.100 --nvidia-ip 192.168.1.101

Or via environment variables:
    SINGULARIS_CYGNUS_IP, SINGULARIS_MACBOOK_IP, SINGULARIS_NVIDIA_IP
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

try:
    import aiohttp
except ImportError:  # pragma: no cover
    print("[ERROR] aiohttp is not installed. Install with: pip install aiohttp")
    sys.exit(1)


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


@dataclass
class ServiceCheck:
    name: str
    host: str
    port: int
    path: str = "/v1/models"
    optional: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"


@dataclass
class ServiceStatus:
    check: ServiceCheck
    ok: bool
    status: Optional[int]
    error: Optional[str]
    body_snippet: str = ""


async def probe_service(session: aiohttp.ClientSession, check: ServiceCheck, timeout: float = 3.0) -> ServiceStatus:
    try:
        async with session.get(check.url, timeout=timeout) as resp:
            text = await resp.text()
            snippet = text[:120].replace("\n", " ")
            ok = 200 <= resp.status < 300
            return ServiceStatus(check=check, ok=ok, status=resp.status, error=None, body_snippet=snippet)
    except asyncio.TimeoutError:
        return ServiceStatus(check=check, ok=False, status=None, error="timeout")
    except Exception as e:  # noqa: BLE001
        return ServiceStatus(check=check, ok=False, status=None, error=str(e))


async def verify_cluster(cygnus_ip: str, macbook_ip: str, nvidia_ip: str) -> int:
    checks: List[ServiceCheck] = []

    # Cygnus: 10 experts on ports 1234–1243
    for port in range(1234, 1244):
        checks.append(ServiceCheck(name=f"Cygnus:{port}", host=cygnus_ip, port=port))

    # MacBook: MoE + AURA
    checks.append(ServiceCheck(name="MacBook:MoE", host=macbook_ip, port=2000))
    # AURA-Brain health is more custom; use /health and mark optional so it doesn't fail the whole cluster
    checks.append(ServiceCheck(name="MacBook:AURA", host=macbook_ip, port=3000, path="/health", optional=True))

    # NVIDIA: Positronic network
    checks.append(ServiceCheck(name="NVIDIA:Positronic", host=nvidia_ip, port=4000, path="/health", optional=True))

    async with aiohttp.ClientSession() as session:
        tasks = [probe_service(session, c) for c in checks]
        results = await asyncio.gather(*tasks)

    # Print summary
    print(f"\n{Colors.HEADER}{Colors.BOLD}SINGULARIS V5.0 CLUSTER HEALTH{Colors.ENDC}")
    print(f"  Cygnus IP  : {cygnus_ip}")
    print(f"  MacBook IP : {macbook_ip}")
    print(f"  NVIDIA IP  : {nvidia_ip}\n")

    cygnus_ok = 0
    cygnus_total = 0
    macbook_ok = 0
    nvidia_ok = 0

    for status in results:
        name = status.check.name
        if status.ok:
            color = Colors.OKGREEN
            state = "UP"
        else:
            color = Colors.FAIL if not status.check.optional else Colors.WARNING
            state = "DOWN" if not status.check.optional else "MISSING/OPTIONAL"

        detail = f"HTTP {status.status}" if status.status is not None else status.error
        print(f"{color}- {name:<18} {state:<16}{Colors.ENDC}  {detail}")

        # Counters
        if name.startswith("Cygnus:"):
            cygnus_total += 1
            if status.ok:
                cygnus_ok += 1
        elif name.startswith("MacBook"):
            if status.ok:
                macbook_ok += 1
        elif name.startswith("NVIDIA"):
            if status.ok:
                nvidia_ok += 1

    # Aggregate summary
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}SUMMARY{Colors.ENDC}")

    # Cygnus experts
    if cygnus_total:
        print(
            f"  Cygnus experts : {cygnus_ok}/{cygnus_total} "
            f"({cygnus_ok / cygnus_total * 100:.1f}% {('UP' if cygnus_ok else 'DOWN')})"
        )

    print(f"  MacBook MoE    : {'UP' if any(s.check.name=='MacBook:MoE' and s.ok for s in results) else 'DOWN'}")
    aura_status = next((s for s in results if s.check.name == 'MacBook:AURA'), None)
    if aura_status:
        print(f"  MacBook AURA   : {'UP' if aura_status.ok else 'MISSING (optional)'}")

    pos_status = next((s for s in results if s.check.name == 'NVIDIA:Positronic'), None)
    if pos_status:
        print(f"  NVIDIA Positr. : {'UP' if pos_status.ok else 'MISSING (optional)'}")

    print("=" * 60 + "\n")

    # Exit code: 0 if all required services are up, 1 otherwise
    required_down = [
        s for s in results
        if not s.check.optional and not s.ok
    ]
    return 0 if not required_down else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Singularis v5 cluster health")
    parser.add_argument("--cygnus-ip", default=os.getenv("SINGULARIS_CYGNUS_IP", "192.168.1.50"))
    parser.add_argument("--macbook-ip", default=os.getenv("SINGULARIS_MACBOOK_IP", "192.168.1.100"))
    parser.add_argument("--nvidia-ip", default=os.getenv("SINGULARIS_NVIDIA_IP", "192.168.1.101"))
    args = parser.parse_args()

    try:
        code = asyncio.run(verify_cluster(args.cygnus_ip, args.macbook_ip, args.nvidia_ip))
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[ABORTED] Cluster verification interrupted by user")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
