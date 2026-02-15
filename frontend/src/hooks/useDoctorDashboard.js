import { useEffect, useMemo, useState } from "react";
import { apiFetch } from "../lib/api.js";

export default function useDoctorDashboard({ authToken, isDoctor }) {
  const [doctorSeniors, setDoctorSeniors] = useState([]);
  const [doctorStats, setDoctorStats] = useState({
    total_seniors: 0,
    total_checkins: 0,
    green: 0,
    yellow: 0,
    red: 0,
    alerts: 0,
    window_days: 7,
  });
  const [doctorLoading, setDoctorLoading] = useState(false);
  const [doctorError, setDoctorError] = useState(null);

  const formattedSeniors = useMemo(() => {
    return doctorSeniors.map((senior) => {
      const lastCheckinAt = senior.lastCheckinAt
        ? new Date(senior.lastCheckinAt).toLocaleString()
        : "No check-in yet";
      const name =
        `${senior.firstName || ""} ${senior.lastName || ""}`.trim() ||
        "Unknown";
      const triage = senior.triageStatus
        ? `${senior.triageStatus}`.replace(/^[a-z]/, (char) =>
            char.toUpperCase(),
          )
        : "—";
      return {
        ...senior,
        displayName: name,
        lastCheckinAt,
        triage,
      };
    });
  }, [doctorSeniors]);

  useEffect(() => {
    const loadDoctorDashboard = async () => {
      if (!isDoctor || !authToken) return;
      setDoctorLoading(true);
      setDoctorError(null);
      try {
        const [analytics, seniors] = await Promise.all([
          apiFetch("/dashboard/analytics", { token: authToken }),
          apiFetch("/dashboard/seniors", { token: authToken }),
        ]);
        setDoctorStats(analytics);
        setDoctorSeniors(seniors?.seniors || []);
      } catch (err) {
        setDoctorError(err?.message || "Failed to load dashboard data");
      } finally {
        setDoctorLoading(false);
      }
    };

    loadDoctorDashboard();
  }, [authToken, isDoctor]);

  return {
    doctorStats,
    formattedSeniors,
    doctorLoading,
    doctorError,
  };
}
