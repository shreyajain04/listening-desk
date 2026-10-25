# Deploy Listening Desk to a public link (about 1 hour, nothing to install)

Everything happens in your browser. You need: a laptop, an email address, and this folder unzipped.

## Part A: Get three accounts and two keys (15 min)

1. **GitHub** (stores the code): go to github.com and sign up.
2. **Groq key** (the main AI model, free):
   - Go to console.groq.com and sign in with Google or email.
   - Open **API Keys**, click **Create API Key**, name it `listening-desk`.
   - Copy the key (it starts with `gsk_`) into a private note. Groq shows it only once.
3. **Gemini key** (free backup model, recommended):
   - Go to aistudio.google.com, sign in, click **Get API key**, then **Create API key**.
   - Copy it into the same private note.

Never paste keys into GitHub, WhatsApp groups or slides. They go only into Streamlit's Secrets box (Part C).

## Part B: Put the code on GitHub (10 min)

1. On GitHub, click **+** (top right), then **New repository**.
2. Name it `listening-desk`, choose **Public**, leave every box unticked, click **Create repository**.
3. On the empty repository page, click the link **uploading an existing file**.
4. Open the unzipped `listening-desk` folder on your laptop. Select everything *inside* it (not the folder itself) and drag it into the browser.
   - **Mac:** press `Cmd + Shift + .` in Finder first. This shows hidden items, so the `.streamlit` folder and `.gitignore` get uploaded too.
   - **Windows:** hidden items are usually visible already.
5. Wait for all files to finish uploading, then click **Commit changes**.
6. Check the repository's front page. You should see `app.py`, `requirements.txt` and the folders `.streamlit`, `core`, `data`, `eval`, `models`, `training`.
   - If `.streamlit` is missing: click **Add file**, then **Create new file**, type `.streamlit/config.toml` as the name, paste the contents of that file from your laptop, and commit. Without it the app still works, but with default colours and fonts.

## Part C: Deploy on Streamlit Community Cloud (10 min, plus a 3-5 min build)

1. Go to share.streamlit.io and click **Continue with GitHub**. Allow access.
2. Click **Create app**, then choose to deploy from a GitHub repository.
3. Fill in:
   - Repository: `your-username/listening-desk`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: pick something short, e.g. `listening-desk-c2`
4. Open **Advanced settings**:
   - Python version: **3.12**
   - Secrets: open `secrets_template.toml`, copy the two key lines, replace the placeholders with your real keys, and paste them in. The names must stay exactly `GROQ_API_KEY` and `GEMINI_API_KEY`.
5. Click **Deploy** and wait for the build to finish.

## Part D: Test it (15 min)

1. **About** tab: click **Test the LLM connection**. Groq should say *Working*. Gemini should too if you added its key.
2. **Inbox** tab: click **Triage 25 mentions**.
   - A red crisis banner should appear.
   - A blue box saying labels come from keyword rules means the keys are not working. Recheck Secrets.
3. Run the four demo cases the brief requires:
   - **Routine mention:** Routine list, open `@meera_travels`. No reply is needed.
   - **Complaint that needs a reply:** Reply list, open `@arjun_m`, click **Draft reply from the playbook**, then **Approve**.
   - **Crisis alert:** Crisis list, open `@fitwithkaran`, click **Acknowledge alert as PR lead**, draft, approve.
   - **Edge case:** **Check a post** tab, click **Sarcasm** or **Hinglish complaint**, then **Check this post**.
4. **Decisions** tab: your approvals should be listed with times.

## If something goes wrong

| What you see | What to do |
|---|---|
| Build fails with a package error | Make sure `requirements.txt` is in the top level of the repository and Python 3.12 was chosen. To change Python, delete the app and deploy again. |
| Error loading the model | The scikit-learn version changed. Keep `scikit-learn==1.8.0` in `requirements.txt`. |
| Labels come from keyword rules | A key is missing, misspelled or out of quota. The About tab's connection test shows the exact error. Edit keys in your app's **Settings**, then **Secrets**. |
| "Rate limit" or slow answers | Groq's free quota is used up for now. The app falls back to Gemini, then to rules. For a demo, the sidebar has **Offline mode**. |
| The app is asleep | Apps sleep after 12 hours without visitors. Open the link and click the button to wake it. Do this an hour before the demo. |

## Changing the app later

Edit any file on GitHub (the pencil icon), then click **Commit changes**. Streamlit redeploys in about a minute. Teammates should test with their own Groq keys and keep the deployed app's key for the app alone, since free limits are per account.
