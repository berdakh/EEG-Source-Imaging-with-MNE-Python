#!/usr/bin/env bash
# Register the standalone EEG-Agent bundle with the desktop (application menu entry + icon).
# Run from anywhere: it uses the bundle directory this script lives in.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$HERE/EEG-Agent"
ICON_SRC="$HERE/_internal/eegagent/assets/icon.png"
[ -x "$EXE" ] || { echo "EEG-Agent executable not found next to this script ($EXE)" >&2; exit 1; }

mkdir -p "$HOME/.local/share/applications" "$HOME/.local/share/icons/hicolor/256x256/apps" "$HOME/.local/bin"
[ -f "$ICON_SRC" ] && cp "$ICON_SRC" "$HOME/.local/share/icons/hicolor/256x256/apps/eegagent.png"
ln -sf "$EXE" "$HOME/.local/bin/eegagent-desktop"
cat > "$HOME/.local/share/applications/eegagent.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=EEG-Agent
Comment=EEG source imaging by conversation
Exec=$EXE
Icon=eegagent
Terminal=false
Categories=Science;Education;
DESKTOP
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$HOME/.local/share/applications" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
echo "EEG-Agent added to your application menu. Remove with: rm ~/.local/share/applications/eegagent.desktop"
