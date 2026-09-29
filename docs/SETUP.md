# ZeroDefect — Access & Environment Setup

Do these once, before milestone M0. Never paste keys or tokens into the chat; store them only in the places described below.

**Needed now:** Steps 1, 5 and 7.
**Before M2:** Steps 2, 3 and 6. **Before M6:** Step 4.

---

## 1. Hugging Face token + Real-IAD access (main dataset)

1. Create a free account at https://huggingface.co/join (skip if you have one).
2. Open https://huggingface.co/datasets/Real-IAD/Real-IAD while logged in.
3. Fill in the short access form at the top of the page and accept the terms (research use only). Approval is automatic.
4. Go to https://huggingface.co/settings/tokens → **Create new token** → token type **Read** → name it `zerodefect` → **Create token**.
5. Copy the token (it starts with `hf_`). You will paste it into the environment settings in Step 5.

## 2. Roboflow API key (PaintDefect dataset)

1. Sign up for free at https://app.roboflow.com.
2. Open your workspace **Settings** → **API Keys** and copy the **Private API Key**.
3. Open https://universe.roboflow.com/ai-klghd/paintdefect-8h4s4 and note the **License** shown on the page. Tell Claude what it says.

## 3. MVTec AD 2 (lighting-robustness dataset)

1. Go to https://www.mvtec.com/company/research/datasets/mvtec-ad-2 → **Downloads**.
2. Fill in the download form and accept the CC BY-NC-SA 4.0 license.
3. Either download the `wallplugs` part yourself (for Colab/Kaggle), or give Claude the download link.
4. For MVTec AD (`screw`, `metal_nut`), no action is needed. The training code downloads it automatically.

## 4. Anthropic API key (AI Copilot, only needed at M6)

1. Sign in at https://platform.claude.com (the Claude Console).
2. Open **API Keys** → **Create Key** → name it `zerodefect-copilot` → copy it (it starts with `sk-ant-`).
3. Open **Settings → Billing** and add prepaid credits. The minimum top-up is $5. API usage is billed separately from a Claude Pro/Max subscription.
4. Store it as **`ZERODEFECT_ANTHROPIC_API_KEY`**, not `ANTHROPIC_API_KEY`. Claude Code itself reads `ANTHROPIC_API_KEY`, so using a different name avoids a clash.

Without this key, the copilot runs in offline mode and answers only from the built-in analytics.

## 5. Configure the cloud environment (network + keys)

1. At https://claude.ai/code, click the **cloud icon with the environment name** in the row above the message box.
2. Hover over your environment (e.g. **Default**) → click the **settings (gear) icon** on the right.
3. Set **Network access** using one of these options:
   - **Option A (easiest): `Full`.** This allows any domain.
   - **Option B (tighter): `Custom`.** Tick **Also include default list of common package managers**, then paste into **Allowed domains**:
     ```
     huggingface.co
     *.huggingface.co
     *.hf.co
     download.pytorch.org
     roboflow.com
     *.roboflow.com
     storage.googleapis.com
     kaggle.com
     *.kaggle.com
     *.kaggleusercontent.com
     mvtec.com
     *.mvtec.com
     *.mydrive.ch
     data.mendeley.com
     ```
     If a download still fails, Claude will tell you which domain to add.
4. In **Environment variables**, add one `KEY=value` per line. Include only the keys you have:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxx
   ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxx
   KAGGLE_API_TOKEN=xxxxxxxxxxxxxxxx
   ZERODEFECT_ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
   ```
   Anyone who uses this environment can read these values. A personal environment is only used by you. On Pro/Max plans, the **API credentials** section lets you store the Hugging Face token so that even Claude never sees it. To do that, set **Allowed websites** to `huggingface.co` and `*.huggingface.co`, set header `Authorization` with prefix `Bearer`, and set the value to your token.
5. Click **Save changes**.
6. **Start a new session.** Running sessions keep the settings they started with. In the new session, say:
   > Continue the ZeroDefect project from `docs/PROJECT_PLAN.md` (branch `claude/beautiful-darwin-pi2ix6`). Start milestone M0.

## 6. GPU for training (free, needed from M2)

The training notebooks will be in `notebooks/`. You can use either Colab or Kaggle.

**Google Colab**
1. Go to https://colab.research.google.com → **File → Open notebook → GitHub** → paste the repo URL → pick the notebook.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Add your tokens in the **🔑 Secrets** panel (left sidebar) using the same names as above.

**Kaggle** (about 30 free GPU hours per week)
1. Verify your phone number under https://www.kaggle.com/settings. This is required for GPU and internet access in notebooks.
2. To get an API token, open the same page → **API** → **Create New Token** → copy it into `KAGGLE_API_TOKEN`.
3. In a notebook: **Settings → Accelerator → GPU T4 x2**, and **Settings → Internet → On**.
4. Add tokens under **Add-ons → Secrets**.

## 7. Academic or commercial?

Reply with one word: **academic** or **commercial**.

- **Academic:** we use the datasets and Ultralytics YOLO as planned.
- **Commercial:** the model gets trained on your own collected images, and we use Apache-licensed components (e.g. RF-DETR) instead of AGPL/non-commercial ones.
