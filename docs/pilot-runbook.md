# Pilot Runbook

## Before install day
- [ ] Office computer specs confirmed (from discovery). Decide: their machine or the appliance.
- [ ] Models pulled on the target machine over a good connection (several GB).
- [ ] Letterhead saved as `packs/church/templates/letterhead.docx`.
- [ ] A past Sunday recording tested through transcription on the same hardware.
- [ ] `.env` written with the chosen models and a PIN.
- [ ] Synthetic samples loaded so the demo works before their documents are in.

## Install day (90 minutes)
1. Run `scripts/install.sh` or `scripts/install.ps1`.
2. Start with the launcher; confirm `/api/health` shows the models installed.
3. Load their bulletins, newsletters, bylaws and policies into the Library.
4. Run `scripts/offline_check.py` with them watching. Then unplug the network and run Sermon Studio.
5. 30-minute training (below). Leave this runbook printed.

## 30-minute training script
- 5 min: what it is, what it never does (nothing leaves this computer, nothing is sent anywhere).
- 10 min: Sermon Studio with last Sunday's recording.
- 5 min: Announcement Hub with this week's real announcements.
- 5 min: Ask the Office with three questions they suggest.
- 5 min: where the files go, how to back up, who to call.

## Weekly check-in (15 minutes, four weeks)
- What did you use it for this week?
- What did it get wrong?
- What did you wish it did?
- Note time saved in their words.

## Exit criteria
See docs/mvp-scope.md, section 5.
