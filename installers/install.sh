#!/usr/bin/env bash
# EEG-Agent desktop installer for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/berdakh/EEG-Source-Imaging-with-MNE-Python/master/installers/install.sh | bash
#
# What it does: finds Python 3.10+, creates a virtual environment in ~/.eegagent/venv,
# installs the package with the desktop extras, and adds a launcher
# (Applications/EEG-Agent.command on macOS, a .desktop entry on Linux).
# Re-run to upgrade. Set EEGAGENT_SOURCE to a local checkout or a git ref to install that instead.
set -euo pipefail

APP_DIR="${EEGAGENT_HOME:-$HOME/.eegagent}"
VENV="$APP_DIR/venv"
SOURCE="${EEGAGENT_SOURCE:-git+https://github.com/berdakh/EEG-Source-Imaging-with-MNE-Python.git@master}"

find_python() {
  for py in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$py"; return 0
    fi
  done
  return 1
}

PY="$(find_python)" || { echo "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ and re-run." >&2; exit 1; }
echo "Using $("$PY" --version) at $(command -v "$PY")"

mkdir -p "$APP_DIR"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
echo "Installing EEG-Agent from $SOURCE (this downloads MNE, SciPy and Streamlit; a few minutes)..."
if [ -d "$SOURCE" ]; then
  "$VENV/bin/python" -m pip install --quiet --upgrade "$SOURCE[desktop]"
else
  "$VENV/bin/python" -m pip install --quiet --upgrade "eeg-source-imaging[desktop] @ $SOURCE"
fi
# pywebview needs a system web view; if it failed to build the app still works in the browser.
"$VENV/bin/python" -c 'import webview' 2>/dev/null || echo "Note: native window unavailable; the app will open in your browser."

LAUNCHER="$VENV/bin/eegagent-desktop"
case "$(uname -s)" in
  Darwin)
    mkdir -p "$HOME/Applications"
    CMD="$HOME/Applications/EEG-Agent.command"
    printf '#!/bin/bash\nexec "%s" "$@"\n' "$LAUNCHER" > "$CMD"
    chmod +x "$CMD"
    echo "Installed. Launch from $CMD (double-click in Finder) or run: $LAUNCHER"
    ;;
  Linux)
    mkdir -p "$HOME/.local/share/applications" "$HOME/.local/bin" "$HOME/.local/share/icons/hicolor/256x256/apps"
    ICON="$("$VENV/bin/python" -c 'import eegagent, os; print(os.path.join(os.path.dirname(eegagent.__file__), "assets", "icon.png"))')"
    [ -f "$ICON" ] && cp "$ICON" "$HOME/.local/share/icons/hicolor/256x256/apps/eegagent.png"
    ln -sf "$LAUNCHER" "$HOME/.local/bin/eegagent-desktop"
    ln -sf "$VENV/bin/eegagent" "$HOME/.local/bin/eegagent"
    cat > "$HOME/.local/share/applications/eegagent.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=EEG-Agent
Comment=EEG source imaging by conversation
Exec=$LAUNCHER
Icon=eegagent
Terminal=false
Categories=Science;Education;
DESKTOP
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$HOME/.local/share/applications" || true
    echo "Installed. Launch 'EEG-Agent' from your application menu or run: eegagent-desktop"
    ;;
  *)
    echo "Installed. Run: $LAUNCHER"
    ;;
esac
echo "Settings (provider, API key, Ollama host) are in the app's sidebar and saved to $APP_DIR/config.json."
