import useAuth from "./hooks/useAuth.js";
import LoginScreen from "./components/LoginScreen.jsx";
import DoctorDashboard from "./components/DoctorDashboard.jsx";
import SeniorCheckin from "./components/SeniorCheckin.jsx";

export default function App() {
  const auth = useAuth();
  const { authUser, authToken, isAuthed, logout } = auth;

  if (!isAuthed) {
    return <LoginScreen auth={auth} />;
  }

  if (authUser?.role === "doctor") {
    return (
      <DoctorDashboard
        authUser={authUser}
        authToken={authToken}
        logout={logout}
      />
    );
  }

  return <SeniorCheckin authUser={authUser} authToken={authToken} logout={logout} />;
}
