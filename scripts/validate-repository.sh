#!/usr/bin/env bash
set -euo pipefail

required=(
  README.md CONTRIBUTING.md SECURITY.md
  .editorconfig .gitattributes .gitignore .env.example
  .github/pull_request_template.md
  .github/ISSUE_TEMPLATE/bug_report.yml
  .github/ISSUE_TEMPLATE/feature_request.yml
)

for file in "${required[@]}"; do
  [[ -s "${file}" ]] || { echo "Missing required file: ${file}" >&2; exit 1; }
done

if git ls-files | grep -E '(^|/)\.env($|\.)' | grep -v -E '(^|/)\.env\.example$' >/dev/null; then
  echo "A local environment file is tracked" >&2
  exit 1
fi

echo "Repository check passed"
