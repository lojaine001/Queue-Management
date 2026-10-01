# VAPID setup for Web Push (Stage 2)

Run this on arnau, in the backend venv. The private key never needs to leave
this machine or pass through chat/git -- generate it here, point an env var
at the file, done.

## 1. Install the dependency (if not already done via requirements.txt)

```powershell
pip install pywebpush
```

This also installs `py-vapid`, which provides the `vapid` CLI used below.

## 2. Generate the keypair

From this folder (`Queue-Management-System-v2-main\Queue-Management-System-v2-main`):

```powershell
vapid --gen
```

This writes two files here: `private_key.pem` and `public_key.pem`. **Do not
commit these** -- they're already covered by this repo's existing
`.gitignore` patterns for `*.pem`/secrets if one exists; if not, add them
explicitly:

```powershell
Add-Content .gitignore "`nprivate_key.pem`npublic_key.pem"
```

## 3. Get the public key string (for the frontend/env var)

```powershell
vapid --applicationServerKey
```

This prints a single base64url string. Copy it.

## 4. Set the three environment variables

Add these to the `.env` file in this same folder (create one if it doesn't
exist yet -- `python-dotenv` already loads it, same as every other script
here):

```
VAPID_PRIVATE_KEY_PATH=C:\Users\arnau\Documents\project\rtsp\Retail-Wait-Prediction\Queue-Management\Queue-Management-System-v2-main\Queue-Management-System-v2-main\private_key.pem
VAPID_PUBLIC_KEY=<the string from step 3>
VAPID_CONTACT_EMAIL=mailto:your-real-email@example.com
```

`VAPID_CONTACT_EMAIL` must be a real address -- it's required by the Web
Push spec as a contact point push services can reach if this server is
ever flagged for abusive traffic. Not a placeholder.

## 5. Restart the API so it picks up `VAPID_PUBLIC_KEY`

The `/vapid-public-key` endpoint reads this at request time from `os.getenv`,
so a plain restart of the `uvicorn` process is enough -- no code change
needed beyond what's already deployed.

## 6. Start the push checker

This is a new, separate long-running process -- same pattern as
`run_scheduler.py`, needs its own terminal window:

```powershell
cd "C:\Users\arnau\Documents\project\rtsp\Retail-Wait-Prediction\Queue-Management\Queue-Management-System-v2-main\Queue-Management-System-v2-main"
.\venv\Scripts\activate
python push_alert_checker.py
```

It checks every 30s by default (`--interval` to change it) and prints a
status line each cycle once there's at least one subscription to check.

## 7. Verify end to end

1. On a test device, open the app, turn on Alertes, click "Enable
   notifications", grant permission.
2. Check the `push_subscriptions` table has a new row:
   ```sql
   SELECT endpoint, threshold_min, horizon_min, was_over FROM push_subscriptions;
   ```
3. Close the browser/app entirely on that device (not just background it --
   fully close it, this is the point of Stage 2).
4. Wait for the wait-time to genuinely cross that device's threshold (or
   temporarily lower the threshold slider to something already exceeded,
   then reopen briefly to let the new threshold sync before closing again).
5. The notification should arrive in the OS tray within one checker cycle
   (30s), with the app fully closed.
