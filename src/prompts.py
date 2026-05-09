"""LLM system prompts for permission-to-policy auditing."""

# Mapping of common Android permissions to plain-language data access description.
# Used to ground the LLM and give it a stable rubric reference.
PERMISSION_GLOSSARY = {
    "android.permission.ACCESS_FINE_LOCATION": "precise GPS location",
    "android.permission.ACCESS_COARSE_LOCATION": "approximate (network-based) location",
    "android.permission.ACCESS_BACKGROUND_LOCATION": "location in the background, even when app not in use",
    "android.permission.CAMERA": "device camera (photos and video)",
    "android.permission.RECORD_AUDIO": "device microphone",
    "android.permission.READ_CONTACTS": "the user's contact list",
    "android.permission.WRITE_CONTACTS": "modify the user's contact list",
    "android.permission.READ_CALENDAR": "calendar events",
    "android.permission.READ_SMS": "SMS messages",
    "android.permission.SEND_SMS": "send SMS messages",
    "android.permission.READ_PHONE_STATE": "phone number, IMEI, network info",
    "android.permission.READ_EXTERNAL_STORAGE": "files on device storage",
    "android.permission.WRITE_EXTERNAL_STORAGE": "write files on device storage",
    "android.permission.READ_MEDIA_IMAGES": "photos and images on device",
    "android.permission.READ_MEDIA_VIDEO": "videos on device",
    "android.permission.READ_MEDIA_AUDIO": "audio files on device",
    "android.permission.BLUETOOTH": "Bluetooth pairing",
    "android.permission.BLUETOOTH_CONNECT": "connect to Bluetooth devices",
    "android.permission.BLUETOOTH_SCAN": "scan for nearby Bluetooth devices",
    "android.permission.ACCESS_WIFI_STATE": "Wi-Fi network info",
    "android.permission.INTERNET": "network communication",
    "android.permission.GET_ACCOUNTS": "device accounts",
    "android.permission.READ_CALL_LOG": "phone call history",
    "android.permission.ACTIVITY_RECOGNITION": "physical activity (walking, running, etc.)",
    "android.permission.BODY_SENSORS": "biometric sensors (heart rate, etc.)",
    "android.permission.POST_NOTIFICATIONS": "show notifications",
    "com.google.android.c2dm.permission.RECEIVE": "push notifications via FCM",
    "com.android.vending.BILLING": "in-app purchases",
}


SYSTEM_PROMPT = """You are a privacy auditor. Your job is to compare an Android app's requested permissions against the substance of its privacy policy and decide, for each permission, whether the policy adequately discloses the corresponding data collection.

For each permission you must produce ONE verdict:
- "covered": the policy clearly states the app collects, uses, or accesses this category of data.
- "unclear": the policy mentions the data category vaguely or only by inference (e.g., generic "device information" for a specific identifier permission).
- "mismatch": the policy does not mention this data category at all, or actively contradicts the permission's purpose.

Strict rules:
1. ALWAYS quote the exact span from the policy that supports a "covered" or "unclear" verdict in the `policy_mention` field. The substring MUST appear verbatim in the source policy.
2. If the policy does not mention the data category, set `policy_mention` to "NOT MENTIONED" and verdict to "mismatch".
3. Do not infer disclosure from app store category, app name, or screenshots — only from the policy text.
4. Reasoning must be one sentence and reference the policy span (or its absence).
5. Return ONLY valid JSON matching the schema, no prose, no markdown fences.

Output schema:
{
  "permissions": [
    {
      "permission": "<full Android permission ID>",
      "data_access": "<plain-language description of what this permission grants>",
      "policy_mention": "<verbatim quoted span from policy, or 'NOT MENTIONED'>",
      "verdict": "covered|unclear|mismatch",
      "reasoning": "<one-sentence justification>"
    }
  ],
  "summary": "<2-3 sentence plain-language summary for a non-technical user>"
}
"""


def build_user_prompt(app_title: str, permissions: list[str], policy_text: str) -> str:
    """Compose the user message for the auditor."""
    glossary_lines = []
    for perm in permissions:
        gloss = PERMISSION_GLOSSARY.get(perm, "(see Android docs)")
        glossary_lines.append(f"- {perm} -> {gloss}")
    glossary = "\n".join(glossary_lines)

    return f"""APP: {app_title}

REQUESTED PERMISSIONS (with grounded data-access reference):
{glossary}

PRIVACY POLICY TEXT:
\"\"\"
{policy_text}
\"\"\"

Audit every permission listed above. Return JSON only."""
