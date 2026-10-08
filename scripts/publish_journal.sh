#!/usr/bin/env bash
# Publish the built journal to the root of the gh-pages branch (https://rozgo.github.io/le-wm-nv/).
# The branch holds only the built site; each publish replaces it and records the source commit.
# Build first: uv run --script scripts/build_journal.py
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SITE="$ROOT/build/journal"
REMOTE=$(git -C "$ROOT" remote get-url origin)
SOURCE=$(git -C "$ROOT" rev-parse --short HEAD)
if [ ! -f "$SITE/index.html" ]; then
    echo "Build the journal first: uv run --script scripts/build_journal.py" >&2
    exit 1
fi
if [ -n "$(git -C "$ROOT" status --porcelain -- site scripts docs)" ]; then
    echo "Commit the journal sources before publishing; the publish records the source commit." >&2
    exit 1
fi
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
if git ls-remote --exit-code --heads "$REMOTE" gh-pages >/dev/null 2>&1; then
    git clone --quiet --depth 1 --branch gh-pages "$REMOTE" "$STAGE"
    find "$STAGE" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
else
    git -C "$STAGE" init --quiet -b gh-pages
    git -C "$STAGE" remote add origin "$REMOTE"
fi
cp -R "$SITE"/. "$STAGE/"
touch "$STAGE/.nojekyll"
git -C "$STAGE" add -A
if git -C "$STAGE" diff --cached --quiet; then
    echo "gh-pages already up to date"
    exit 0
fi
git -C "$STAGE" -c user.name="$(git -C "$ROOT" config user.name)" -c user.email="$(git -C "$ROOT" config user.email)" \
    commit --quiet -m "Publish the journal from $SOURCE"
git -C "$STAGE" push --quiet origin gh-pages
echo "published the journal from $SOURCE"
