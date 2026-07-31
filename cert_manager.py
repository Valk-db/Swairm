"""
Certificate Manager for FCS Anchor.
Handles TLS certificates: self-signed for dev, Let's Encrypt (ACME) for production.
"""

import os
import ssl
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Any
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# Try to import ACME libraries (optional)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import josepy as jose
    from acme import client, messages, challenges, crypto_util
    from acme.client import ClientV2
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

try:
    import josepy as jose
    from acme import client, messages, challenges, crypto_util
    from acme.client import ClientV2
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    ACME_AVAILABLE = True
except ImportError:
    ACME_AVAILABLE = False
    # Runtime stubs (never actually used since we check ACME_AVAILABLE before calling)
    jose = None  # type: ignore
    client = None  # type: ignore
    messages = None  # type: ignore
    challenges = None  # type: ignore
    crypto_util = None  # type: ignore
    ClientV2 = None  # type: ignore
    x509 = None  # type: ignore
    NameOID = None  # type: ignore
    hashes = None  # type: ignore
    serialization = None  # type: ignore
    rsa = None  # type: ignore


class CertManager:
    """
    Manages TLS certificates for the Anchor server.

    Supports three modes:
    1. Self-signed (dev mode, default)
    2. Let's Encrypt via ACME (production, requires domain + port 80/443)
    3. Pre-existing cert files (manual)
    """

    def __init__(
        self,
        cert_dir: Path,
        domain: Optional[str] = None,
        email: Optional[str] = None,
        staging: bool = False,
        auto_renew: bool = True,
    ):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.email = email
        self.staging = staging
        self.auto_renew = auto_renew

        self.cert_path = self.cert_dir / "cert.pem"
        self.key_path = self.cert_dir / "key.pem"
        self.chain_path = self.cert_dir / "chain.pem"
        self.fullchain_path = self.cert_dir / "fullchain.pem"

        self._renewal_task: Optional[asyncio.Task] = None
        # ACME HTTP-01 challenge storage: token -> key_authorization
        self._acme_challenges: dict[str, str] = {}

    def register_acme_challenge(self, token: str, key_authorization: str) -> None:
        """Register an ACME HTTP-01 challenge response for Let's Encrypt validation."""
        self._acme_challenges[token] = key_authorization

    def get_acme_challenge(self, token: str) -> Optional[str]:
        """Get the ACME challenge response for a given token."""
        return self._acme_challenges.get(token)

    def clear_acme_challenge(self, token: str) -> None:
        """Clear an ACME challenge after validation."""
        self._acme_challenges.pop(token, None)

    def has_valid_cert(self) -> bool:
        """Check if we have a valid (non-expired) certificate."""
        if not self.cert_path.exists() or not self.key_path.exists():
            return False
        try:
            # ssl._ssl._test_decode_cert is a private API but works across Python versions
            cert = ssl._ssl._test_decode_cert(str(self.cert_path))  # type: ignore[attr-defined]
            not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
            return not_after > datetime.now(timezone.utc) + timedelta(days=7)  # 7-day buffer
        except Exception:
            return False

    def generate_self_signed(
        self,
        common_name: str = "localhost",
        days: int = 365,
        key_size: int = 2048,
    ) -> Tuple[Path, Path]:
        """Generate a self-signed certificate for development."""
        logger.info(f"Generating self-signed certificate for {common_name}...")

        if ACME_AVAILABLE:
            # Use cryptography library
            key = rsa.generate_private_key(  # type: ignore[union-attr]
                public_exponent=65537,
                key_size=key_size,
            )

            subject = issuer = x509.Name([  # type: ignore[union-attr]
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),  # type: ignore[union-attr]
            ])
            cert = (
                x509.CertificateBuilder()  # type: ignore[union-attr]
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())  # type: ignore[union-attr]
                .not_valid_before(datetime.utcnow())
                .not_valid_after(datetime.utcnow() + timedelta(days=days))
                .add_extension(
                    x509.SubjectAlternativeName([  # type: ignore[union-attr]
                        x509.DNSName(common_name),  # type: ignore[union-attr]
                        x509.DNSName("localhost"),  # type: ignore[union-attr]
                        x509.IPAddress(__import__('ipaddress').ip_address("127.0.0.1")),  # type: ignore[union-attr]
                    ]),
                    critical=False,
                )
                .sign(key, hashes.SHA256())  # type: ignore[union-attr]
            )
            cert_pem = cert.public_bytes(serialization.Encoding.PEM)  # type: ignore[union-attr]
            key_pem = key.private_bytes(
                encoding=serialization.Encoding.PEM,  # type: ignore[union-attr]
                format=serialization.PrivateFormat.PKCS8,  # type: ignore[union-attr]
                encryption_algorithm=serialization.NoEncryption(),  # type: ignore[union-attr]
            )
        else:
            # Fallback to openssl command
            cert_pem, key_pem = self._gen_self_signed_openssl(common_name, days, key_size)

        self.cert_path.write_bytes(cert_pem)
        self.key_path.write_bytes(key_pem)

        # For self-signed, chain = cert
        self.chain_path.write_bytes(cert_pem)
        self.fullchain_path.write_bytes(cert_pem)

        logger.info(f"Self-signed cert saved to {self.cert_dir}")
        return self.cert_path, self.key_path

    def _gen_self_signed_openssl(self, cn: str, days: int, key_size: int) -> Tuple[bytes, bytes]:
        """Generate self-signed cert using openssl command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            key_file = tmpdir / "key.pem"
            csr_file = tmpdir / "csr.pem"
            cert_file = tmpdir / "cert.pem"

            # Generate key
            subprocess.run(["openssl", "genrsa", "-out", str(key_file), str(key_size)], check=True)
            # Generate CSR
            subprocess.run([
                "openssl", "req", "-new", "-key", str(key_file), "-out", str(csr_file),
                "-subj", f"/CN={cn}"
            ], check=True)
            # Self-sign
            subprocess.run([
                "openssl", "x509", "-req", "-in", str(csr_file), "-signkey", str(key_file),
                "-out", str(cert_file), "-days", str(days)
            ], check=True)

            cert_pem = cert_file.read_bytes()
            key_pem = key_file.read_bytes()

        return cert_pem, key_pem

    async def request_letsencrypt_cert(self) -> Tuple[Path, Path]:
        """Request a certificate from Let's Encrypt via ACME (HTTP-01 challenge)."""
        if not ACME_AVAILABLE:
            raise RuntimeError(
                "ACME libraries not installed. Run: pip install acme josepy cryptography"
            )
        if not self.domain:
            raise ValueError("Domain required for Let's Encrypt certificate")
        if not self.email:
            raise ValueError("Email required for Let's Encrypt registration")

        logger.info(f"Requesting Let's Encrypt certificate for {self.domain}...")

        # Run the synchronous ACME operations in a thread pool
        return await asyncio.to_thread(self._request_letsencrypt_cert_sync)

    def _request_letsencrypt_cert_sync(self) -> Tuple[Path, Path]:
        """Synchronous implementation of Let's Encrypt certificate request."""
        # ACME directory URL
        directory_url = (
            "https://acme-staging-v02.api.letsencrypt.org/directory"
            if self.staging
            else "https://acme-v02.api.letsencrypt.org/directory"
        )

        # Create or load account key
        account_key_path = self.cert_dir / "account_key.pem"
        if account_key_path.exists():
            with open(account_key_path, "rb") as f:
                account_key = jose.JWKRSA.load(f.read())  # type: ignore[union-attr]
        else:
            account_key = jose.JWKRSA(key=rsa.generate_private_key(  # type: ignore[union-attr]
                public_exponent=65537, key_size=2048
            ))
            account_key_path.write_bytes(account_key.to_pem())  # type: ignore[union-attr]

        # Create ACME client
        net = client.ClientNetwork(account_key, user_agent="FCS-Anchor/1.0")  # type: ignore[union-attr]
        directory = messages.Directory.from_json(net.get(directory_url).json())  # type: ignore[union-attr]
        acme_client = ClientV2(directory, net=net)  # type: ignore[union-attr]

        # Register account
        try:
            regr = acme_client.new_account(
                messages.NewRegistration.from_data(  # type: ignore[union-attr]
                    email=self.email, terms_of_service_agreed=True
                )
            )
            logger.info(f"ACME account registered: {regr.uri}")
        except messages.ConflictError:  # type: ignore[union-attr]
            pass  # Account already exists

        # Request certificate
        cert_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)  # type: ignore[union-attr]
        csr = crypto_util.make_csr(cert_key, [self.domain])  # type: ignore[union-attr]

        order = acme_client.new_order(csr)
        authz = order.authorizations[0]  # type: ignore[index]

        # HTTP-01 challenge (requires port 80)
        challenge = next(
            c for c in authz.body.challenges
            if isinstance(c.chall, challenges.HTTP01)  # type: ignore[union-attr]
        )

        response = challenge.response(account_key)
        # Register the challenge response so the HTTP endpoint can serve it
        self.register_acme_challenge(challenge.chall.token, response.key_authorization)  # type: ignore[union-attr]
        acme_client.answer_challenge(challenge, response)

        # Wait for validation
        authz = acme_client.poll(authz)
        if authz.body.status != messages.STATUS_VALID:  # type: ignore[union-attr, attr-defined]
            raise RuntimeError(f"Challenge failed: {authz.body.status}")  # type: ignore[union-attr, attr-defined]

        # Clear the challenge after successful validation
        self.clear_acme_challenge(challenge.chall.token)  # type: ignore[union-attr]

        # Finalize order
        order = acme_client.poll_and_finalize(order)
        fullchain_pem = order.fullchain_pem

        # Save certificate and key
        self.cert_path.write_bytes(crypto_util.dump_certificate(order.certificate).encode())  # type: ignore[union-attr]
        self.key_path.write_bytes(cert_key.private_bytes(
            encoding=serialization.Encoding.PEM,  # type: ignore[union-attr]
            format=serialization.PrivateFormat.PKCS8,  # type: ignore[union-attr]
            encryption_algorithm=serialization.NoEncryption(),  # type: ignore[union-attr]
        ))

        # Save chain
        self.chain_path.write_bytes(
            "\n".join(crypto_util.dump_certificate(c).decode() for c in order.chain).encode()  # type: ignore[union-attr]
        )
        self.fullchain_path.write_bytes(fullchain_pem.encode())

        logger.info(f"Let's Encrypt cert saved to {self.cert_dir}")
        return self.cert_path, self.key_path

    def load_existing_cert(self, cert_file: Path, key_file: Path) -> Tuple[Path, Path]:
        """Load pre-existing certificate and key files."""
        import shutil
        shutil.copy2(cert_file, self.cert_path)
        shutil.copy2(key_file, self.key_path)

        # Try to load chain if available
        chain_file = cert_file.parent / "chain.pem"
        fullchain_file = cert_file.parent / "fullchain.pem"
        if chain_file.exists():
            shutil.copy2(chain_file, self.chain_path)
        else:
            shutil.copy2(cert_file, self.chain_path)

        if fullchain_file.exists():
            shutil.copy2(fullchain_file, self.fullchain_path)
        else:
            # Create fullchain from cert + chain
            cert_data = self.cert_path.read_bytes()
            chain_data = self.chain_path.read_bytes()
            self.fullchain_path.write_bytes(cert_data + b"\n" + chain_data)

        logger.info(f"Loaded existing cert from {cert_file}")
        return self.cert_path, self.key_path

    async def ensure_certificate(self) -> Tuple[Path, Path]:
        """Ensure we have a valid certificate, generating/requesting as needed."""
        if self.has_valid_cert():
            logger.info(f"Using existing valid certificate from {self.cert_dir}")
            return self.cert_path, self.key_path

        # Priority: Let's Encrypt (if domain configured) > self-signed
        if self.domain and self.email and ACME_AVAILABLE:
            try:
                return await self.request_letsencrypt_cert()
            except Exception as e:
                logger.warning(f"Let's Encrypt failed: {e}. Falling back to self-signed.")

        # Fallback: self-signed
        cn = self.domain or "localhost"
        return self.generate_self_signed(common_name=cn)

    async def start_auto_renewal(self):
        """Start background task for certificate renewal."""
        if not self.auto_renew or not self.domain:
            return

        async def renewal_loop():
            while True:
                # Check every 12 hours
                await asyncio.sleep(12 * 3600)
                if not self.has_valid_cert():
                    logger.info("Certificate expiring soon, attempting renewal...")
                    try:
                        await self.request_letsencrypt_cert()
                        logger.info("Certificate renewed successfully")
                    except Exception as e:
                        logger.error(f"Renewal failed: {e}")

        self._renewal_task = asyncio.create_task(renewal_loop())
        logger.info("Auto-renewal started")

    def stop_auto_renewal(self):
        """Stop the auto-renewal background task."""
        if self._renewal_task:
            self._renewal_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._renewal_task)
            except asyncio.CancelledError:
                pass


def create_cert_manager_from_env(cert_dir: Path) -> CertManager:
    """Create CertManager from environment variables."""
    return CertManager(
        cert_dir=cert_dir,
        domain=os.environ.get("FCS_TLS_DOMAIN"),
        email=os.environ.get("FCS_TLS_EMAIL"),
        staging=os.environ.get("FCS_TLS_STAGING", "false").lower() == "true",
        auto_renew=os.environ.get("FCS_TLS_AUTO_RENEW", "true").lower() == "true",
    )