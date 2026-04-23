"use client";

import { useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { clsx } from "@/lib/clsx";
import fathyLogo from "@/app/fathy.png";

export default function LoginClient() {
  const router = useRouter();
  const { login, isLoading } = useAuth();
  const [error, setError] = useState<string>("");
  const [formData, setFormData] = useState({ email: "", password: "" });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");

    try {
      await login(formData.email, formData.password);
      router.push("/chat");
    } catch (err: any) {
      setError(err.message || "Login failed. Please try again.");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[rgb(var(--bg))]">
      <div className="w-full max-w-sm p-8 bg-[rgb(var(--card))] rounded-2xl border border-[rgb(var(--border))] shadow-sm">
        <div className="flex flex-col items-center gap-3 mb-8">
          <Image src={fathyLogo} alt="Fathy" width={48} height={48} className="rounded-xl object-contain" />
          <h1 className="text-xl font-semibold text-[rgb(var(--fg))]">Welcome back</h1>
          <p className="text-sm text-[rgb(var(--muted))]">Log in to your account</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm dark:bg-red-950/30 dark:border-red-900/50 dark:text-red-400">
              {error}
            </div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-[rgb(var(--fg))] mb-1">
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              className="w-full px-3 py-2.5 rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--input-bg))] text-[rgb(var(--fg))] text-sm placeholder-[rgb(var(--muted))] focus:outline-none focus:border-[rgb(var(--fg))]/40 transition"
              placeholder="your@email.com"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-[rgb(var(--fg))] mb-1">
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              className="w-full px-3 py-2.5 rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--input-bg))] text-[rgb(var(--fg))] text-sm placeholder-[rgb(var(--muted))] focus:outline-none focus:border-[rgb(var(--fg))]/40 transition"
              placeholder="••••••••"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={clsx(
              "w-full py-2.5 rounded-lg bg-[rgb(var(--fg))] text-[rgb(var(--bg))] text-sm font-medium hover:opacity-80 transition disabled:opacity-40",
              isLoading && "cursor-not-allowed"
            )}
          >
            {isLoading ? "Logging in..." : "Log In"}
          </button>
        </form>

        <p className="mt-6 text-center text-sm text-[rgb(var(--muted))]">
          Don't have an account?{" "}
          <Link href="/auth/signup" className="text-[rgb(var(--fg))] font-medium underline hover:opacity-70">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
