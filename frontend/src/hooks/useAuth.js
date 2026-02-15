import { useEffect, useState } from "react";
import * as Auth from "../lib/auth.js";

const TOKEN_STORAGE_KEY = "guardian_checkin.jwt";

export default function useAuth() {
  const [authMode, setAuthMode] = useState("login");
  const [authFirstName, setAuthFirstName] = useState("");
  const [authLastName, setAuthLastName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authUser, setAuthUser] = useState(null);
  const [authToken, setAuthToken] = useState(() => {
    try {
      return localStorage.getItem(TOKEN_STORAGE_KEY);
    } catch {
      return null;
    }
  });
  const [authStatus, setAuthStatus] = useState("idle");
  const [authError, setAuthError] = useState(null);
  const isAuthed = Boolean(authToken && authUser?.email);

  const persistToken = (token) => {
    setAuthToken(token);
    try {
      if (token) localStorage.setItem(TOKEN_STORAGE_KEY, token);
      else localStorage.removeItem(TOKEN_STORAGE_KEY);
    } catch {
      // ignore
    }
  };

  const refreshMe = async (token) => {
    if (!token) {
      setAuthUser(null);
      return;
    }
    try {
      const user = await Auth.me({ token });
      setAuthUser(user);
    } catch (err) {
      persistToken(null);
      setAuthUser(null);
    }
  };

  useEffect(() => {
    refreshMe(authToken);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submitAuth = async (event) => {
    event?.preventDefault?.();
    setAuthError(null);
    setAuthStatus("loading");
    const firstName = authFirstName.trim();
    const lastName = authLastName.trim();
    const email = authEmail.trim().toLowerCase();
    const password = authPassword;

    try {
      if (!email || !password) {
        throw new Error("Email and password are required");
      }

      if (authMode === "register") {
        if (!firstName || !lastName) {
          throw new Error("First name and last name are required");
        }
        if (!email.includes("@gmail.com")) {
          throw new Error("Please use a valid Gmail address");
        }
        await Auth.register({ firstName, lastName, email, password });
      }

      const { access_token } = await Auth.login({ email, password });
      persistToken(access_token);
      await refreshMe(access_token);
      setAuthPassword("");
      setAuthFirstName("");
      setAuthLastName("");
    } catch (err) {
      setAuthError(err?.message || "Auth failed");
    } finally {
      setAuthStatus("idle");
    }
  };

  const logout = () => {
    persistToken(null);
    setAuthUser(null);
    setAuthError(null);
  };

  return {
    authMode,
    setAuthMode,
    authFirstName,
    setAuthFirstName,
    authLastName,
    setAuthLastName,
    authEmail,
    setAuthEmail,
    authPassword,
    setAuthPassword,
    authUser,
    authToken,
    authStatus,
    authError,
    isAuthed,
    refreshMe,
    submitAuth,
    logout,
  };
}
