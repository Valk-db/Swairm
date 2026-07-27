"""
Certificate Manager for FCS Anchor.
Handles TLS certificates: self-signed for dev, Let's Encrypt (ACME) for production.
"""

import os
import ssl
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# Try to import ACME libraries (optional)
try:
    import josepy as jose
    from acme import client, messages, challenges, crypto_util
    from acme.client import ClientV2
    ACME_AVAILABLE = True
except ImportError:
    ACME_AVAILABLE = False
    ClientV2 = None
    messages = None
    challenges = None
    crypto_util = None
    jose = None


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

    def has_valid_cert(self) -> bool:
        """Check if we have a valid (non-expired) certificate."""
        if not self.cert_path.exists() or not self.key_path.exists():
            return False
        try:
            cert = ssl._ssl._test_decode_cert(str(self.cert_path))
            not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
            return not_after > datetime.utcnow() + timedelta(days=7)  # 7-day buffer
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

        # Generate private key
        key = crypto_util.make_key(key_size) if ACME_AVAILABLE else self._gen_key_openssl(key_size)

        # Generate self-signed cert
        if ACME_AVAILABLE:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization

            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ])
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(datetime.utcnow() + timedelta(days=days))
                .add_extension(
                    x509.SubjectAlternativeName([
                        x509.DNSName(common_name),
                        x509.DNSName("localhost"),
                        x509.IPAddress(__import__('ipaddress').ip_address("127.0.0.1")),
                    ]),
                    critical=False,
                )
                .sign(key, hashes.SHA256())
            )
            cert_pem = cert.public_bytes(serialization.Encoding.PEM)
            key_pem = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
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

    def _gen_key_openssl(self, key_size: int):
        """Generate RSA key using openssl command."""
        result = subprocess.run(
            ["openssl", "genrsa", str(key_size)],
            capture_output=True, check=True
        )
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        return serialization.load_pem_private_key(result.stdout, password=None)

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
        """Request a certificate from Let's Encrypt via ACME."""
        if not ACME_AVAILABLE:
            raise RuntimeError("ACME libraries not installed. Run: pip install acme josepy cryptography")
        if not self.domain:
            raise ValueError("Domain required for Let's Encrypt certificate")
        if not self.email:
            raise ValueError("Email required for Let's Encrypt registration")

        logger.info(f"Requesting Let's Encrypt certificate for {self.domain}...")

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
                account_key = jose.JWKRSA.load(f.read())
        else:
            account_key = jose.JWKRSA(key=crypto_util.make_key(2048))
            account_key_path.write_bytes(account_key.to_pem())

        # Create ACME client
        net = client.ClientNetwork(account_key, user_agent="FCS-Anchor/1.0")
        directory = messages.Directory.from_json(net.get(directory_url).json())
        acme_client = ClientV2(directory, net=net)

        # Register account
        try:
            regr = acme_client.new_account(
                messages.NewRegistration.from_data(email=self.email, terms_of_service_agreed=True)
            )
            logger.info(f"ACME account registered: {regr.uri}")
        except messages.ConflictError:
            # Account already exists
            pass

        # Request certificate
        cert_key = crypto_util.make_key(2048)
        csr = crypto_util.make_csr(cert_key, [self.domain])

        # HTTP-01 challenge (requires port 80)
        order = acme_client.new_order(csr)
        authz = order.authorizations[0]
        challenge = next(c for c in authz.body.challenges if isinstance(c.chall, challenges.HTTP01))

        # TODO: This requires a running HTTP server on port 80 to serve the challenge
        # For now, we'll note this limitation and use TLS-ALPN-01 if available
        # or suggest using certbot externally

        response = challenge.response(account_key)
        acme_client.answer_challenge(challenge, response)

        # Wait for validation
        authz = acme_client.poll(authz)
        if authz.body.status != messages.STATUS_VALID:
            raise RuntimeError(f"Challenge failed: {authz.body.status}")

        # Finalize order
        order = acme_client.poll_and_finalize(order)
        fullchain_pem = order.fullchain_pem

        # Save certificate and key
        self.cert_path.write_bytes(crypto_util.dump_certificate(order.certificate).encode())
        self.key_path.write_bytes(cert_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode())

        # Save chain and fullchain
        chain_pem = "\n".join(crypto_util.dump_certificate(c).decode() for c in order.chain)
        self.chain_path.write_bytes(chain_pem.encode())
        self.fullchain_path.write_bytes(fullchain_pem.encode())

        logger.info(f"Let's Encrypt certificate saved to {self.cert_dir}")
        return self.cert_path, self.key_path

    async def ensure_certificate(self) -> Tuple[Path, Path]:
        """
        Ensure we have a valid certificate. Strategy:
        1. If valid cert exists, use it
        2. If domain configured and ACME available, try Let's Encrypt
        3. Fall back to self-signed
        """
        if self.has_valid_cert():
            logger.info(f"Using existing valid certificate from {self.cert_dir}")
            return self.cert_path, self.key_path

        if self.domain and ACME_AVAILABLE:
            try:
                return await self.request_letsencrypt_cert()
            except Exception as e:
                logger.warning(f"Let's Encrypt failed: {e}. Falling back to self-signed.")

        # Fallback: self-signed
        cn = self.domain or "localhost"
        return self.generate_self_signed(common_name=cn)

    def get_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for uvicorn."""
        cert_path, key_path = asyncio.run(self.ensure_certificate())

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert_path), str(key_path))
        # Modern security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")
        return context

    async def start_auto_renewal(self, check_interval_hours: int = 12):
        """Start background task for certificate renewal."""
        if not self.auto_renew or not self.domain:
            return

        async def renewal_loop():
            while True:
                await asyncio.sleep(check_interval_hours * 3600)
                if self.has_valid_cert():
                    # Check if expiring soon (within 30 days)
                    try:
                        cert = ssl._ssl._test_decode_cert(str(self.cert_path))
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        if not_after < datetime.utcnow() + timedelta(days=30):
                            logger.info("Certificate expiring soon, attempting renewal...")
                            try:
                                await self.request_letsencrypt_cert()
                                logger.info("Certificate renewed successfully")
                            except Exception as e:
                                logger.error(f"Certificate renewal failed: {e}")
                    except Exception:
                        pass

        self._renewal_task = asyncio.create_task(renewal_loop())
        logger.info("Certificate auto-renewal started")

    def stop_auto_renewal(self):
        if self._renewal_task:
            self._renewal_task.cancel()
            self._renewal_task = None


def create_cert_manager_from_env(cert_dir: Path) -> CertManager:
    """Create CertManager from environment variables."""
    return CertManager(
        cert_dir=cert_dir,
        domain=os.environ.get("FCS_TLS_DOMAIN"),
        email=os.environ.get("FCS_TLS_EMAIL"),
        staging=os.environ.get("FCS_TLS_STAGING", "false").lower() == "true",
        auto_renew=os.environ.get("FCS_TLS_AUTO_RENEW", "true").lower() == "true",
    )