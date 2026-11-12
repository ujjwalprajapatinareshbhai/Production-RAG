import getpass
import sys

from app.auth import hash_password
from app.registry import get_user_by_email
from app.db import get_connection


TENANT_SLUG = "legacy-workspace"


def main():
    print("\n=== Change Owner Password ===\n")

    email = input("Owner email: ").strip().lower()

    user = get_user_by_email(email)

    if user is None:
        print("User not found.")
        sys.exit(1)

    new_password = getpass.getpass("Enter new password: ")
    confirm_password = getpass.getpass("Confirm new password: ")

    if len(new_password) < 12:
        print("Password must contain at least 12 characters.")
        sys.exit(1)

    if new_password != confirm_password:
        print("Passwords do not match.")
        sys.exit(1)

    password_hash = hash_password(new_password)

    with get_connection() as conn:
        with conn.cursor() as cursor:

            # Verify owner role and active account.
            cursor.execute(
                """
                SELECT
                    tm.role,
                    u.is_active
                FROM tenant_memberships tm
                JOIN tenants t
                    ON t.tenant_id = tm.tenant_id
                JOIN users u
                    ON u.user_id = tm.user_id
                WHERE tm.user_id = %s
                  AND t.slug = %s
                  AND tm.is_active = TRUE
                FOR UPDATE OF tm, u;
                """,
                (user["user_id"], TENANT_SLUG),
            )

            membership = cursor.fetchone()

            if membership is None:
                print("User has no active membership in this tenant.")
                sys.exit(1)

            role, is_active = membership

            if role != "owner":
                print("Error: Only the tenant owner can use this script.")
                sys.exit(1)

            if not is_active:
                print("Error: The owner account is inactive.")
                sys.exit(1)

            # Update the verified owner's password.
            cursor.execute(
                """
                UPDATE users
                SET password_hash = %s
                WHERE user_id = %s;
                """,
                (password_hash, user["user_id"]),
            )

    print("\nOwner password changed successfully.")


if __name__ == "__main__":
    main()