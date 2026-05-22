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

The processor inherits :py:class:`OnlineProcessor`'s full error / timeout /
SSL / captcha / metrics handling.

.. note::

   Parallel pagination assumes pages are independently fetchable. Engines that
   require a token from page 1 to request page 2 (e.g. DuckDuckGo's ``vqd``)
   cannot use this processor without a custom :py:func:`requests` implementation.
"""

__all__ = ["OnlineFanoutProcessor"]

import copy
import typing as t

import searx.network
from searx.metrics.error_recorder import count_error

from .online import OnlineProcessor, OnlineParams

if t.TYPE_CHECKING:
    from searx.result_types import EngineResults


# Default fan-out width if the engine module doesn't set ``fanout_pages``.
DEFAULT_FANOUT_PAGES = 4


class OnlineFanoutProcessor(OnlineProcessor):
    """Processor for engines that fan out one user query into many parallel HTTP requests."""

    engine_type: str = "online_fanout"

    def _build_request_list(self, query: str, params: OnlineParams) -> list[searx.network.Request]:
        """Return the list of sub-requests for one user query.

        Dispatches on the engine's API:

        - If the engine defines ``requests(query, params)`` → use it directly.
        - Else fall back to ``request(query, params_for_page)`` looped over
          ``pageno = 1..fanout_pages``, building one :py:class:`network.Request`
          per page from the populated params.
        """
        engine_requests_fn = getattr(self.engine, "requests", None)
        if callable(engine_requests_fn):
            return list(engine_requests_fn(query, params))

        fanout_pages: int = int(getattr(self.engine, "fanout_pages", DEFAULT_FANOUT_PAGES))
        if fanout_pages < 1:
            fanout_pages = 1

        request_list: list[searx.network.Request] = []
        for pageno in range(1, fanout_pages + 1):
            page_params: OnlineParams = copy.deepcopy(params)  # type: ignore[assignment]
            page_params["pageno"] = pageno  # type: ignore[typeddict-item]

            # Standard engines populate ``params`` in place: url, method, headers,
            # cookies, data/json/content, etc.
            self.engine.request(query, page_params)

            if not page_params.get("url"):
                continue

            request_list.append(self._params_to_request(page_params))

        return request_list

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
        request_list = self._build_request_list(query, params)
        if not request_list:
            return None

        responses = searx.network.multi_requests(request_list)

        merged: list = []
        for sub_request, resp in zip(request_list, responses):
            if isinstance(resp, Exception):
                count_error(
                    self.engine.name,
                    "fanout subrequest failed: {0}".format(resp.__class__.__name__),
                    (sub_request.method, sub_request.url),
                    secondary=True,
                )
                continue

            # ``response()`` reads ``search_params`` off the httpx.Response —
            # match the contract used by the standard online flow.
            resp.search_params = params  # type: ignore[attr-defined]
            sub_results = self.engine.response(resp)
            if sub_results:
                merged.extend(sub_results)

        return merged or None


# expose the default for engine modules and tests that want to refer to it
__all__.append("DEFAULT_FANOUT_PAGES")
