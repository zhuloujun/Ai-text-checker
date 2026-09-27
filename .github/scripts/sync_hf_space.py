"""把仓库里的 hf-space/ 文件夹同步到 Hugging Face Space（由 GitHub Actions 调用）。

- Space 不存在时自动创建（Docker 类型、免费 CPU）
- 如果提供了 HF_ADMIN_TOKEN，同步写入 Space 的 Secret：ADMIN_TOKEN
- 远端有、本地没有的文件会被删除（.gitattributes 除外），保证两边完全一致

环境变量：
  HF_TOKEN        必填，Hugging Face 的 Access Token（需要 write 权限）
  HF_SPACE        Space 名称，形如 用户名/space名，默认 tdyso/ai-text-checker
  HF_ADMIN_TOKEN  可选，Space 管理页密码
  GIT_SHA         可选，写进提交说明
"""
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.utils import HfHubHTTPError

SRC = Path(__file__).resolve().parents[2] / "hf-space"
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    space = os.environ.get("HF_SPACE", "").strip() or "tdyso/ai-text-checker"
    if not token:
        sys.exit("缺少 HF_TOKEN：请在 GitHub 仓库 Settings → Secrets and variables → Actions 里添加。")
    api = HfApi(token=token)

    try:
        who = api.whoami()
    except HfHubHTTPError as e:
        sys.exit(f"HF_TOKEN 无效或已过期：{e}")
    print(f"已登录 Hugging Face：{who.get('name')}")

    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space：https://huggingface.co/spaces/{space}")

    admin = os.environ.get("HF_ADMIN_TOKEN", "").strip()
    if admin:
        api.add_space_secret(space, "ADMIN_TOKEN", admin)
        print("已写入 Space Secret：ADMIN_TOKEN")
    else:
        print("未提供 HF_ADMIN_TOKEN：请确认 Space 里已手动设置 ADMIN_TOKEN，否则无法签发 API Key。")

    local = {}
    for p in SRC.rglob("*"):
        if p.is_file() and not (set(p.relative_to(SRC).parts) & SKIP_DIRS):
            local[p.relative_to(SRC).as_posix()] = p
    remote = set(api.list_repo_files(space, repo_type="space"))

    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in sorted(local.items())]
    ops += [CommitOperationDelete(path_in_repo=f) for f in sorted(remote - set(local)) if f != ".gitattributes"]
    sha = os.environ.get("GIT_SHA", "")[:7]
    info = api.create_commit(space, repo_type="space", operations=ops,
                             commit_message=f"Sync from GitHub {sha}".strip())
    print(f"已同步 {len(local)} 个文件，删除 {len(ops) - len(local)} 个旧文件：{info.commit_url}")
    print(f"Space 会自动重新构建（约 5–15 分钟）。网页：https://{space.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
