"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { api, ApiError, tokenManager, type User } from "@/lib/api";

type AuthContextType = {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<User>;
  register: (email: string, username: string, password: string) => Promise<User>;
  logout: () => Promise<void>;
  deleteAccount: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize from stored token on mount
  useEffect(() => {
    const initializeAuth = async () => {
      const token = tokenManager.getToken();
      if (token) {
        try {
          const currentUser = await api.auth.getCurrentUser();
          setUser(currentUser);
        } catch (error) {
          console.error("Failed to get current user:", error);
          const status = error instanceof ApiError ? error.status : null;
          if (status === 401 || status === 403) {
            tokenManager.clearToken();
          }
          setUser(null);
        }
      }
      setIsLoading(false);
    };

    initializeAuth();
  }, []);

  useEffect(() => {
    const handleAuthTokenChanged = () => {
      if (!tokenManager.getToken()) {
        setUser(null);
      }
    };

    window.addEventListener("auth-token-changed", handleAuthTokenChanged);
    return () => window.removeEventListener("auth-token-changed", handleAuthTokenChanged);
  }, []);

  const login = async (email: string, password: string): Promise<User> => {
    setIsLoading(true);
    try {
      const response = await api.auth.login({ email, password });
      tokenManager.setToken(response.access_token);
      setUser(response.user);
      return response.user;
    } catch (error) {
      tokenManager.clearToken();
      setUser(null);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (email: string, username: string, password: string): Promise<User> => {
    setIsLoading(true);
    try {
      const response = await api.auth.register({ email, username, password });
      tokenManager.setToken(response.access_token);
      setUser(response.user);
      return response.user;
    } catch (error) {
      tokenManager.clearToken();
      setUser(null);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async (): Promise<void> => {
    setIsLoading(true);
    try {
      await api.auth.logout();
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      tokenManager.clearToken();
      setUser(null);
      setIsLoading(false);
    }
  };

  const deleteAccount = async (): Promise<void> => {
    setIsLoading(true);
    try {
      await api.auth.deleteAccount();
      tokenManager.clearToken();
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
        deleteAccount
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
