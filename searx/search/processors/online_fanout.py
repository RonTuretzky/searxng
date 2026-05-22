# SPDX-License-Identifier: AGPL-3.0-or-later
"""Processor used for ``online_fanout`` engines.

An ``online_fanout`` engine fans a single user query out into N parallel HTTP
requests, dispatched via :py:func:`searx.network.multi_requests`. Two engine
APIs are supported:

1. **Standard online API** (``request(query, params)`` + ``response(resp)``).
   The processor reuses an existing engine module (e.g. ``google``,
   ``bing``) and loops ``pageno = 1..fanout_pages`` to build N requests in
   parallel. Set ``engine_type: online_fanout`` and ``fanout_pages: N`` in
   ``settings.yml`` for any paging-capable engine to get parallel pagination.

2. **Fan-out native API** (``requests(query, params)`` returning
   ``list[network.Request]`` + ``response(resp)``). For engines that need
   custom fan-out semantics (sub-queries, sharded backends, multi-provider
   aggregation) rather than parallel pagination.

Error policy
~~~~~~~~~~~~

:py:func:`searx.network.multi_requests` returns exceptions in the response
list rather than raising them. To preserve the parent
:py:class:`OnlineProcessor.search` flow — which handles SSL / timeout /
captcha / rate-limit suspension — this processor:

- treats single sub-request failures as *partial* failures: increments a
  secondary error counter and keeps the surviving results, and
- re-raises the first sub-request exception when **every** sub-request
  failed (so the engine looks fully broken to the base handler, which can
  then suspend it).

.. note::

   Parallel pagination assumes pages are independently fetchable. Engines that
   require a token from page 1 to request page 2 (e.g. DuckDuckGo's ``vqd``)
   cannot use this processor without a custom :py:func:`requests` implementation.
"""

__all__ = ["OnlineFanoutProcessor", "DEFAULT_FANOUT_PAGES"]

import copy
import typing as t

import searx.network
from searx.metrics.error_recorder import count_error

from .online import OnlineProcessor, OnlineParams

if t.TYPE_CHECKING:
    from searx.result_types import EngineResults


# Default fan-out width if the engine module doesn't set ``fanout_pages``.
DEFAULT_FANOUT_PAGES = 4


# A pair of (Request descriptor, params dict that produced it). Tracking the
# params per request lets each response see its own per-page context — required
# for engines that read pageno / time_range / etc. off ``resp.search_params``
# in their ``response()`` parser.
_SubRequest = tuple[searx.network.Request, "OnlineParams"]


class OnlineFanoutProcessor(OnlineProcessor):
    """Processor for engines that fan out one user query into many parallel HTTP requests."""

    engine_type: str = "online_fanout"

    def _build_request_list(self, query: str, params: OnlineParams) -> list[_SubRequest]:
        """Return the list of (sub-request, per-request params) pairs.

        Dispatches on the engine's API:

        - If the engine defines ``requests(query, params)`` → use it directly.
          All sub-requests share the same outer ``params`` dict (the engine
          decided to fan out itself, so it owns context per request).
        - Else fall back to ``request(query, page_params)`` looped over
          ``pageno = 1..fanout_pages``, building one :py:class:`network.Request`
          per page from the populated params and remembering the per-page
          params so each response can be parsed with the right context.
        """
        engine_requests_fn = getattr(self.engine, "requests", None)
        if callable(engine_requests_fn):
            return [(req, params) for req in engine_requests_fn(query, params)]

        fanout_pages: int = int(getattr(self.engine, "fanout_pages", DEFAULT_FANOUT_PAGES))
        if fanout_pages < 1:
            fanout_pages = 1

        sub_requests: list[_SubRequest] = []
        for pageno in range(1, fanout_pages + 1):
            page_params: OnlineParams = copy.deepcopy(params)  # type: ignore[assignment]
            page_params["pageno"] = pageno  # type: ignore[typeddict-item]

            # Standard engines populate ``params`` in place: url, method, headers,
            # cookies, data/json/content, etc.
            self.engine.request(query, page_params)

            if not page_params.get("url"):
                continue

            sub_requests.append((self._params_to_request(page_params), page_params))

        return sub_requests

    @staticmethod
    def _params_to_request(params: OnlineParams) -> searx.network.Request:
        """Convert a populated ``OnlineParams`` dict to a :py:class:`network.Request`.

        Mirrors the request-args assembly in
        :py:meth:`OnlineProcessor._send_http_request` so that engines wrapped by
        ``online_fanout`` see the same HTTP semantics they would under
        ``online``.
        """
        kwargs: dict[str, t.Any] = {
            "headers": params["headers"],
            "cookies": params["cookies"],
            "auth": params["auth"],
            "raise_for_httperror": params.get("raise_for_httperror", True),
        }

        verify = params.get("verify")
        if verify is not None:
            kwargs["verify"] = verify

        max_redirects = params.get("max_redirects")
        if max_redirects:
            kwargs["max_redirects"] = max_redirects

        if "allow_redirects" in params:
            kwargs["allow_redirects"] = params["allow_redirects"]

        method = params["method"]
        if method == "POST":
            if params["data"]:
                kwargs["data"] = params["data"]
            if params["json"]:
                kwargs["json"] = params["json"]
            if params["content"]:
                kwargs["content"] = params["content"]

        return searx.network.Request(method, t.cast(str, params["url"]), kwargs)

    def _search_basic(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, query: str, params: OnlineParams
    ) -> "EngineResults | list | None":
        sub_requests = self._build_request_list(query, params)
        if not sub_requests:
            return None

        request_list = [req for req, _ in sub_requests]
        responses = searx.network.multi_requests(request_list)

        merged: list = []
        first_exception: BaseException | None = None
        for (sub_request, sub_params), resp in zip(sub_requests, responses):
            if isinstance(resp, Exception):
                if first_exception is None:
                    first_exception = resp
                count_error(
                    self.engine.name,
                    "fanout subrequest failed: {0}".format(resp.__class__.__name__),
                    (sub_request.method, sub_request.url),
                    secondary=True,
                )
                continue

            # ``response()`` reads ``search_params`` off the httpx.Response —
            # use the per-request params so the engine sees the right pageno
            # (and any other per-page context) for THIS response.
            resp.search_params = sub_params  # type: ignore[attr-defined]
            sub_results = self.engine.response(resp)
            if sub_results:
                merged.extend(sub_results)

        # If nothing came back AND every sub-request raised, propagate so the
        # parent OnlineProcessor.search() can apply its SSL / timeout / captcha
        # / rate-limit suspension policy. Partial failures (some successes) are
        # treated as a degraded-but-OK engine — the secondary error counters
        # above record them without suspending the engine.
        if not merged and first_exception is not None:
            raise first_exception

        return merged or None
