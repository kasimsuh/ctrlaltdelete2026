# MongoDB Schema — Guardian Check-In (Hackathon MVP)

**Simplified for speed**: Only 2 collections: `users` and `checkin_history`

**Database Name**: `guardian`

---

## Collections

| Collection | Purpose |
|-----------|---------|
| `users` | All accounts (seniors, doctors/admins) |
| `checkin_history` | Check-in records with metrics & triage results |

---

## 1. Users Collection

```javascript
{
  _id: ObjectId("..."),
  email: "john.doe@example.com",
  password_hash: "$pbkdf2-sha256$...",
  
  // Profile
  name: "John Doe",
  role: "senior", // or "doctor"
  
  // For seniors: emergency contact
  emergency_contact_name: "Jane Doe",
  emergency_contact_phone: "+1-555-0101",
  
  // For doctors: can see dashboard of all seniors
  is_admin: role === "doctor", // Auto-determined
  
  // Meta
  created_at: ISODate("2026-02-14T10:00:00Z"),
  is_active: true
}
```

**Indexes**:
```javascript
db.users.createIndex({ email: 1 }, { unique: true })
db.users.createIndex({ role: 1 })
```

---

## 2. Checkin History Collection

```javascript
{
  _id: ObjectId("..."),
  
  // Links
  user_id: ObjectId("..."), // Reference to users._id (the senior)
  
  // Session IDs
  screening_session_id: "screening_1771116551516",
  checkin_id: "5c1c3483-5486-47f5-b96a-693b74c0f954",
  
  // Timing
  started_at: ISODate("2026-02-15T00:49:11Z"),
  completed_at: ISODate("2026-02-15T00:50:21Z"),
  
  // Status & Triage
  status: "completed", // or "in_progress", "failed"
  triage_status: "green", // or "yellow", "red"
  triage_reasons: [
    "Facial asymmetry metrics are within normal range",
    "No symptoms reported (no dizziness, chest pain, or trouble breathing)",
    "Medication compliance confirmed"
  ],
  
  // Self-report flags (from Q&A screening)
  answers: {
    dizziness: false,
    chest_pain: false,
    trouble_breathing: false,
    medication_taken: true
  },
  
  // Facial symmetry analysis (from face module)
  metrics: {
    facial_symmetry: {
      mouth: {
        median_percent: 3.52,
        p90_percent: 4.16,
        level: "normal" // "normal", "warn", "alert"
      },
      eye: {
        median_percent: 1.06,
        p90_percent: 2.63,
        level: "normal"
      },
      nasolabial: {
        median_percent: 5.59,
        p90_percent: 6.64,
        level: "normal"
      },
      combined_index: 0.29, // 0-1 scale
      quality: {
        valid_frames: 289,
        total_frames: 289,
        quality_ratio: 1.0,
        duration_seconds: 10.9,
        index_mean: 0.34063694761341196,
        index_std: 1.6070674024446703
      }
    }
  },
  
  // Screening Q&A responses (from Gemini)
  screening_responses: [
    {
      question_index: 0,
      question: "How are you feeling today?",
      answer: true, // parsed boolean
      transcript: "how are you feeling today I'm feeling good"
    },
    {
      question_index: 1,
      question: "Are you experiencing any dizziness, chest pain, or trouble breathing?",
      answer: false,
      transcript: "no I am not"
    },
    {
      question_index: 2,
      question: "Did you take your morning medications?",
      answer: true,
      transcript: "yes I did"
    }
  ],
  
  // Full transcript (concatenated from all Q&A)
  transcript: "AI: How are you feeling today? USER: how are you feeling today I'm feeling good AI: Are you experiencing any dizziness, chest pain, or trouble breathing? USER: no I am not AI: Did you take your morning medications? USER: yes I did",
  
  // User-facing message
  user_message: "Great check-in! All metrics are normal and you're reporting feeling well.",
  
  // For clinician/doctor dashboard
  clinician_notes: "Day 42: All facial symmetry metrics within normal range. Medication compliance confirmed. No acute concerns.",
  
  // Alert escalation (if triggered)
  alert_level: null, // or "yellow", "red"
  alert_sent: false,
  alert_target: null, // phone or email if alert needed
  alert_message: null,
  alert_sent_at: null,
  
  created_at: ISODate("2026-02-15T00:49:11Z")
}
```

**Indexes**:
```javascript
db.checkin_history.createIndex({ user_id: 1, completed_at: -1 })
db.checkin_history.createIndex({ triage_status: 1, completed_at: -1 })
db.checkin_history.createIndex({ created_at: -1 })
```

---

## Key Queries

### For Seniors

1. **Get recent check-ins**:
   ```javascript
   db.checkin_history.find({ user_id: ObjectId("...") })
     .sort({ completed_at: -1 })
     .limit(10)
   ```

### For Doctors/Admins

1. **Get all seniors**:
   ```javascript
   db.users.find({ role: "senior" })
   ```

2. **Get recent check-ins for all seniors**:
   ```javascript
   db.checkin_history.find({ status: "completed", triage_status: { $in: ["yellow", "red"] } })
     .sort({ completed_at: -1 })
     .limit(20)
   ```

3. **Get check-in history for a specific senior**:
   ```javascript
   db.checkin_history.find({ user_id: ObjectId("...") })
     .sort({ completed_at: -1 })
   ```

---

## Data Validation (Optional)

**users**:
```javascript
db.users.schema = {
  bsonType: "object",
  required: ["email", "password_hash", "name", "role", "created_at"],
  properties: {
    role: { enum: ["senior", "doctor"] }
  }
}
```

**checkin_history**:
```javascript
db.checkin_history.schema = {
  bsonType: "object",
  required: ["user_id", "started_at", "status", "triage_status"],
  properties: {
    status: { enum: ["in_progress", "completed", "failed"] },
    triage_status: { enum: ["green", "yellow", "red"] }
  }
}
```

---

## PyMongo Examples

```python
from pymongo import MongoClient
from datetime import datetime, timedelta

client = MongoClient(os.environ.get("MONGODB_URI"))
db = client["guardian"]

# Register user
db.users.insert_one({
  "email": "john@example.com",
  "password_hash": hash_password("password123"),
  "name": "John Doe",
  "role": "senior",
  "emergency_contact_name": "Jane Doe",
  "emergency_contact_phone": "+1-555-0101",
  "created_at": datetime.utcnow(),
  "is_active": True
})

# Store check-in (with face asymmetry + screening data)
db.checkin_history.insert_one({
  "user_id": ObjectId("..."),
  "screening_session_id": "screening_1771116551516",
  "checkin_id": "5c1c3483-5486-47f5-b96a-693b74c0f954",
  "started_at": datetime(2026, 2, 15, 0, 49, 11, tzinfo=timezone.utc),
  "completed_at": datetime(2026, 2, 15, 0, 50, 21, tzinfo=timezone.utc),
  "status": "completed",
  "triage_status": "green",
  "triage_reasons": [
    "Facial asymmetry metrics are within normal range",
    "No symptoms reported",
    "Medication compliance confirmed"
  ],
  "answers": {
    "dizziness": False,
    "chest_pain": False,
    "trouble_breathing": False,
    "medication_taken": True
  },
  "metrics": {
    "facial_symmetry": {
      "mouth": {"median_percent": 3.52, "p90_percent": 4.16, "level": "normal"},
      "eye": {"median_percent": 1.06, "p90_percent": 2.63, "level": "normal"},
      "nasolabial": {"median_percent": 5.59, "p90_percent": 6.64, "level": "normal"},
      "combined_index": 0.29,
      "quality": {
        "valid_frames": 289,
        "total_frames": 289,
        "quality_ratio": 1.0,
        "duration_seconds": 10.9,
        "index_mean": 0.34063694761341196,
        "index_std": 1.6070674024446703
      }
    }
  },
  "screening_responses": [
    {
      "question_index": 0,
      "question": "How are you feeling today?",
      "answer": True,
      "transcript": "how are you feeling today I'm feeling good"
    },
    {
      "question_index": 1,
      "question": "Are you experiencing any dizziness, chest pain, or trouble breathing?",
      "answer": False,
      "transcript": "no I am not"
    },
    {
      "question_index": 2,
      "question": "Did you take your morning medications?",
      "answer": True,
      "transcript": "yes I did"
    }
  ],
  "transcript": "AI: How are you feeling today? USER: I'm feeling good...",
  "user_message": "Great check-in! All metrics are normal.",
  "clinician_notes": "All facial symmetry metrics within normal range. Medication compliance confirmed.",
  "alert_level": None,
  "alert_sent": False,
  "created_at": datetime.utcnow()
})

# Get recent check-ins for senior
user_id = ObjectId("...")
recent = list(db.checkin_history.find({ "user_id": user_id }).sort("completed_at", -1).limit(10))

# Dashboard: All red/yellow alerts
alerts = list(db.checkin_history.find(
  { "triage_status": { "$in": ["yellow", "red"] } }
).sort("completed_at", -1).limit(50))

# Get all seniors
seniors = list(db.users.find({ "role": "senior" }))
```

---

## Summary

- **users**: Email, password hash, name, role (senior/doctor), emergency contact
- **checkin_history**: Check-in record, metrics, triage result, alert status
- **Indexes**: For fast lookups by user + date, and triage filtering
- **Access**: Doctors see all seniors + all check-ins; seniors see their own

**Ready to wire up FastAPI!**
