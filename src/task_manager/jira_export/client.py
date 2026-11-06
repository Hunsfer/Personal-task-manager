"""Jira Cloud REST API. Error messages hold status codes and field names, never task text."""

from pathlib import Path
from typing import Any

import httpx

REQUEST_TIMEOUT_S = 120.0


class JiraError(Exception):
    """Base for everything Jira can answer with."""


class JiraAuthError(JiraError):
    """401 or 403: usually an expired token. Nothing can be sent until the owner replaces it."""


class JiraUnavailableError(JiraError):
    """Jira can't be reached or is failing. Worth trying again later."""


class JiraRateLimitedError(JiraError):
    def __init__(self, seconds: float) -> None:
        super().__init__(f"rate limited for {seconds} seconds")
        self.seconds = seconds


class JiraNotFoundError(JiraError):
    """404: the task or sprint is gone."""


class JiraRejectedError(JiraError):
    """Jira refused this request, so repeating it unchanged won't help."""

    def __init__(self, status: int, fields: list[str]) -> None:
        super().__init__(f"HTTP {status}: {','.join(fields) or 'no fields'}")
        self.status = status
        self.fields = fields


class JiraClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=httpx.BasicAuth(email, api_token),
            timeout=REQUEST_TIMEOUT_S,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise JiraUnavailableError(type(error).__name__) from None
        status = response.status_code
        if status < 300:
            return response
        if status in (401, 403):
            raise JiraAuthError(f"HTTP {status}")
        if status == 404:
            raise JiraNotFoundError(path.split("/")[-1])
        if status == 429:
            raise JiraRateLimitedError(float(response.headers.get("Retry-After", 60)))
        if status >= 500:
            raise JiraUnavailableError(f"HTTP {status}")
        raise JiraRejectedError(status, _error_fields(response))

    async def myself(self) -> str:
        return (await self._request("GET", "/rest/api/3/myself")).json()["accountId"]

    async def project_statuses(self, project_key: str) -> dict[str, list[str]]:
        """Each issue type of the project with the statuses it can have."""
        response = await self._request("GET", f"/rest/api/3/project/{project_key}/statuses")
        return {
            item["name"]: [status["name"] for status in item.get("statuses", [])]
            for item in response.json()
        }

    async def find_my_recent_task(self, project_key: str, title: str) -> str | None:
        """A task of ours with exactly this title, from the last few days.

        The title is compared here rather than in the query, because Jira's text search is fuzzy.
        The old /search was removed, and this one returns no fields unless they're asked for.
        """
        response = await self._request(
            "POST",
            "/rest/api/3/search/jql",
            json={
                "jql": (
                    f'project = "{project_key}" AND creator = currentUser() '
                    "AND created >= -3d ORDER BY created DESC"
                ),
                "fields": ["summary"],
                "maxResults": 50,
            },
        )
        for issue in response.json().get("issues", []):
            if issue["fields"]["summary"] == title:
                return issue["key"]
        return None

    async def create_issue(self, fields: dict[str, Any]) -> str:
        response = await self._request("POST", "/rest/api/3/issue", json={"fields": fields})
        return response.json()["key"]

    async def issue_state(self, key: str) -> tuple[str, dict[str, Any] | None]:
        """The task's status name and its description document."""
        response = await self._request(
            "GET", f"/rest/api/3/issue/{key}", params={"fields": "status,description"}
        )
        fields = response.json()["fields"]
        return fields["status"]["name"], fields.get("description")

    async def update_description(self, key: str, description: dict[str, Any]) -> None:
        await self._request(
            "PUT", f"/rest/api/3/issue/{key}", json={"fields": {"description": description}}
        )

    async def link_issues(self, link_type: str, new_key: str, original_key: str) -> None:
        await self._request(
            "POST",
            "/rest/api/3/issueLink",
            json={
                "type": {"name": link_type},
                "inwardIssue": {"key": new_key},
                "outwardIssue": {"key": original_key},
            },
        )

    async def upload_attachment(self, key: str, path: Path, name: str) -> None:
        with path.open("rb") as file:
            await self._request(
                "POST",
                f"/rest/api/3/issue/{key}/attachments",
                headers={"X-Atlassian-Token": "no-check"},
                files={"file": (name, file)},
            )


def _error_fields(response: httpx.Response) -> list[str]:
    """Which fields Jira complained about. Its messages may quote the task, so they're dropped."""
    try:
        errors = response.json().get("errors", {})
    except ValueError:
        return []
    return sorted(errors) if isinstance(errors, dict) else []
