import getpass
import sys

from app.auth import hash_password
from app.registry import bootstrap_first_owner


TENANT_SLUG = "legacy-workspace"


def main() -> None:
    print("\n=== Production RAG: First Owner Bootstrap ===\n")

    email = input("Owner email: ").strip().lower()

    if not email or "@" not in email:
        print("Error: Enter a valid email address.")
        sys.exit(1)

    password = getpass.getpass("Owner password: ")
    confirm_password = getpass.getpass("Confirm password: ")

    if len(password) < 12:
        print("Error: Password must contain at least 12 characters.")
        sys.exit(1)

    if password != confirm_password:
        print("Error: Passwords do not match.")
        sys.exit(1)

    try:
        password_hash = hash_password(password)

        result = bootstrap_first_owner(
            tenant_slug=TENANT_SLUG,
            email=email,
            password_hash=password_hash,
        )

        print("\nFirst owner created successfully.")
        print(f"Tenant: {result['tenant']['name']}")
        print(f"Email: {result['user']['email']}")
        print(f"Role: {result['membership']['role']}")
        print(f"User ID: {result['user']['user_id']}")

    except Exception as exc:
        print(f"\nBootstrap failed: {exc}")
        sys.exit(1)

    finally:
        password = None
        confirm_password = None
        password_hash = None


if __name__ == "__main__":
    main()