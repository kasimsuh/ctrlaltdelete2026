/* eslint-disable no-undef */
// Runs once on first container startup (when the data directory is empty).
// Creates an app-scoped DB user that the backend can authenticate with.

const appDb = process.env.MONGO_APP_DB || "guardian";
const appUser = process.env.MONGO_APP_USER || "guardian_app";
const appPass = process.env.MONGO_APP_PASS || "guardian_app_pass";

print(`[mongo-init] creating app user '${appUser}' on db '${appDb}'`);

const dbRef = db.getSiblingDB(appDb);

const existing = dbRef.getUser(appUser);
if (existing) {
  print(`[mongo-init] user '${appUser}' already exists, skipping`);
} else {
  dbRef.createUser({
    user: appUser,
    pwd: appPass,
    roles: [{ role: "readWrite", db: appDb }],
  });
  print(`[mongo-init] user '${appUser}' created`);
}

