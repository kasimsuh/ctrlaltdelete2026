import { apiFetch } from "./api.js";

export function register({ email, password, firstName, lastName }) {
  return apiFetch("/auth/register", { method: "POST", body: { email, password, firstName, lastName } });
}

export function login({ email, password }) {
  return apiFetch("/auth/login", { method: "POST", body: { email, password } });
}

export function me({ token }) {
  return apiFetch("/me", { method: "GET", token });
}

