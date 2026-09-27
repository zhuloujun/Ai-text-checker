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
    name = who.get("name") if isinstance(who, dict) else getattr(who, "name", "?")
    print(f"已登录 Hugging Face：{name}")
    owner = space.split("/")[0]
    orgs = [o.get("name") for o in (who.get("orgs") or [])] if isinstance(who, dict) else []
    if owner != name and owner not in orgs:
        sys.exit(f"令牌属于账号 {name}，但目标 Space 在 {owner} 名下。请用 {owner} 账号生成令牌，"
                 f"或在 GitHub 变量 HF_SPACE 里改成 {name}/ai-text-checker。")

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


def fail(msg: str):
    # ::error:: 会显示在 GitHub Actions 页面的错误摘要里
    print(f"::error::{msg}", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if isinstance(e.code, str):
            fail(e.code)
        raise
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", "?")
        hint = {401: "令牌无效或已过期，请重新生成 HF_TOKEN。",
                403: "令牌没有写权限：生成令牌时类型要选 Write；如果 Space 属于组织，令牌账号要有该组织的写权限。",
                402: "Hugging Face 自 2026 年 7 月起，免费账号不能在 CPU 上托管 Docker Space，需要开通 PRO（https://huggingface.co/pro）后再运行本工作流。",
                404: "找不到 Space 或账号，请检查 HF_SPACE 名称。"}.get(status, "")
        fail(f"Hugging Face 返回错误（HTTP {status}）：{hint} 详情：{str(e)[:400]}")
    except Exception as e:  # noqa: BLE001
        fail(f"{type(e).__name__}: {str(e)[:500]}")
