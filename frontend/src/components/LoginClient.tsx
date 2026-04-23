"use client";

import { useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import Link from "next/link";
import fathyLogo from "@/app/fathy.png";
import { useAuth } from "@/lib/auth-context";
import { clsx } from "@/lib/clsx";

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
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#0f0f1a] via-[#1a1035] to-[#0f0f1a]">
      <div className="w-full max-w-md p-8 bg-white/5 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/10">
        <div className="text-center mb-8">
          <div className="flex flex-col items-center gap-3">
            <Image src={fathyLogo} alt="Fathy Logo" width={64} height={64} className="rounded-2xl object-contain" />
            <h1 className="text-3xl font-bold text-white">Fathy</h1>
            <p className="text-slate-400">Your AI Assistant</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded text-red-200 text-sm">
              {error}
            </div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-1">
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              className="w-full px-4 py-3 bg-white/8 border border-white/15 rounded-xl text-white placeholder-white/30 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition"
              placeholder="your@email.com"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-slate-300 mb-1">
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              className="w-full px-4 py-3 bg-white/8 border border-white/15 rounded-xl text-white placeholder-white/30 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition"
              placeholder="••••••••"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={clsx(
              "w-full py-3 px-4 rounded-xl font-semibold bg-gradient-to-r from-violet-600 to-purple-600 text-white hover:from-violet-700 hover:to-purple-700 transition-all shadow-lg disabled:opacity-50",
              isLoading && "cursor-not-allowed"
            )}
          >
            {isLoading ? "Logging in..." : "Log In"}
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-slate-400 text-sm">
            Don't have an account?{" "}
            <Link href="/auth/signup" className="text-violet-300 hover:text-violet-200 font-medium">
              Sign Up
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
