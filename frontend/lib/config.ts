const value = process.env.NEXT_PUBLIC_API_BASE_URL;

if (!value) {
  throw new Error(
    "NEXT_PUBLIC_API_BASE_URL is not set. Copy .env.local.example to .env.local and set it.",
  );
}

export const API_BASE = value;
