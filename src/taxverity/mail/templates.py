"""Plain-text bodies for the account-flow emails. Kept as code, not a
template engine — three short messages do not need one."""

from __future__ import annotations


def verify_email_body(link: str) -> str:
    return (
        "Welcome to TaxVerity.\n\n"
        "Confirm this address to finish creating your account:\n"
        f"{link}\n\n"
        "This link expires in 24 hours. If you did not request this, "
        "ignore this email."
    )


def already_registered_body(reset_link: str) -> str:
    return (
        "You already have a TaxVerity account with this email address.\n\n"
        "If this was you and you forgot your password, reset it here:\n"
        f"{reset_link}\n\n"
        "If you did not try to register, no action is needed."
    )


def reset_password_body(link: str) -> str:
    return (
        "A password reset was requested for your TaxVerity account.\n\n"
        f"{link}\n\n"
        "This link expires in 30 minutes and can be used once. If you did "
        "not request this, ignore this email — your password will not "
        "change."
    )
